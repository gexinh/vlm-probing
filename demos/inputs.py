"""Native model/dataset input contracts shared by the portable notebooks."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from vlm_probing import ProbeInputs, TokenLayout
SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")
ANSWER_INSTRUCTION = " \nAnswer the question using a single word or phrase."
def load_samples(directory):
    directory = Path(directory)
    rows = json.loads((directory / "samples.json").read_text())
    for row in rows:
        image_path = directory / row["image_path"]
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if list(image.size) != row["image_size"]:
            raise ValueError("GQA image dimensions do not match author annotation")
        x, y, w, h = row["bbox_xywh"]
        if not (0 <= x < x + w <= image.width and 0 <= y < y + h <= image.height):
            raise ValueError("Bounding box lies outside the original image")
    return rows

def layer_window(center, layer_count, width=9):
    """Author rule: [center - width//2, center + width//2], clipped."""
    if type(width) is not int or width < 1 or width % 2 == 0:
        raise ValueError("Window width must be a positive odd integer")
    if type(layer_count) is not int or layer_count < 1:
        raise ValueError("layer_count must be positive")
    if type(center) is not int or not 0 <= center < layer_count:
        raise ValueError("Window center is outside the decoder")
    radius = width // 2
    return list(range(max(0, center - radius), min(layer_count, center + radius + 1)))

def pad_square(image, image_mean):
    """LLaVA author's ``image_aspect_ratio='pad'`` preprocessing."""
    image = image.convert("RGB")
    side = max(image.size)
    left, top = (side - image.width) // 2, (side - image.height) // 2
    fill = tuple(int(channel * 255) for channel in image_mean)
    padded = Image.new("RGB", (side, side), fill)
    padded.paste(image, (left, top))
    return padded, (left, top)

def target_patch_mask(image, bbox_xywh, processor, *, patch_size=14):
    """Author's red-mask rule through the *same* resize/crop as the image.

    A patch is a target patch if at least one processed pixel is exactly red.
    This follows ``InformationFlow.py`` rather than approximating a box in an
    unpadded image. The remainder is the complement over all visual tokens,
    including the square-pad tokens in LLaVA-1.5's fixed 24×24 grid.
    """
    x, y, w, h = bbox_xywh
    mask = Image.new("RGB", image.size, (0, 255, 0))
    ImageDraw.Draw(mask).rectangle((x, y, x + w, y + h), fill=(255, 0, 0))
    padded, offset = pad_square(mask, processor.image_processor.image_mean)
    pixels = processor.image_processor(
        images=padded, do_normalize=False, do_rescale=False, return_tensors="pt"
    )["pixel_values"][0]
    height, width = pixels.shape[-2:]
    if height != width or height % patch_size:
        raise ValueError("Expected a square CLIP input divisible by patch_size")
    red = (pixels == torch.tensor([255, 0, 0])[:, None, None]).all(0)
    grid = red.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size).any(-1).any(-1)
    if not grid.any():
        raise ValueError("No target patches survived native image preprocessing")
    return grid.reshape(-1), {"padding_offset_xy": list(offset), "encoder_size": [width, height],
                              "patch_size": patch_size, "grid_shape": list(grid.shape),
                              "region_rule": "any exactly red processed bbox-mask pixel",
                              "other_rule": "all visual patches except target; includes square padding"}

class FixedAnswerProbability:
    """Whole-vocabulary probability of one fixed token at the answer position."""
    def __init__(self, token_id):
        self.token_id = int(token_id)
        self.last_readout = None

    def score(self, logits, layout):
        selected = layout.select("last_prompt")
        if not (selected.sum(-1) == 1).all():
            raise ValueError("Exactly one answer-prediction position is required")
        probabilities = logits[selected].float().softmax(-1)
        top_probability, top_id = probabilities.max(-1)
        self.last_readout = {"token_id": top_id.detach().cpu().tolist(),
                             "probability": top_probability.detach().cpu().tolist()}
        return probabilities[:, self.token_id]

def prepare_case(sample, data_dir, processor, device, *, dtype=torch.bfloat16):
    """Native HF processor input with verified expanded image/question masks."""
    image = Image.open(Path(data_dir) / sample["image_path"]).convert("RGB")
    padded, offset = pad_square(image, processor.image_processor.image_mean)
    prompt = SYSTEM_PROMPT + " USER: <image>\n" + sample["question"] + ANSWER_INSTRUCTION + " ASSISTANT:"
    values = dict(processor(text=prompt, images=padded, return_tensors="pt"))
    ids = values["input_ids"]
    visual = ids == processor.image_token_id
    patch_size = processor.patch_size
    target_grid, geometry = target_patch_mask(image, sample["bbox_xywh"], processor, patch_size=patch_size)
    if int(visual.sum()) != target_grid.numel():
        raise ValueError("Expanded visual token count does not match the encoder patch grid")
    expanded = prompt.replace(processor.image_token, processor.image_token * int(visual.sum()))
    offset_tokens = processor.tokenizer(expanded, return_offsets_mapping=True, return_tensors="pt")
    if not torch.equal(ids, offset_tokens["input_ids"]):
        raise ValueError("Offset-tokenized input disagrees with native LlavaProcessor input")
    q_start = expanded.index(sample["question"])
    q_end = q_start + len(sample["question"])
    offsets = offset_tokens["offset_mapping"]
    question = (offsets[..., 1] > q_start) & (offsets[..., 0] < q_end)
    question &= ~visual
    target = torch.zeros_like(visual)
    target[visual] = target_grid
    last = torch.zeros_like(visual)
    last[:, -1] = True
    valid = values["attention_mask"].bool()
    if not question.any() or (question & visual).any() or (question & last).any():
        raise ValueError("Question, image, and final prediction positions must be distinct")
    masks = {"image": visual, "question": question, "last": last,
             "target": target, "other": visual & ~target}
    if not torch.equal(masks["target"] | masks["other"], visual):
        raise ValueError("Target and other patches must partition visual positions")
    info = {"prompt": prompt, "question": sample["question"], "input_token_count": ids.shape[1],
            "image_token_count": int(visual.sum()), "target_patch_count": int(target.sum()),
            "other_patch_count": int(masks["other"].sum()), "geometry": geometry,
            "token_positions": {name: torch.where(mask[0])[0].tolist() for name, mask in masks.items()},
            "question_decoded": processor.tokenizer.decode(ids[question].tolist()),
            "last_position": int(ids.shape[1] - 1), "last_token_id": int(ids[0, -1]),
            "last_token_text": processor.tokenizer.decode(ids[0, -1:].tolist()),
            "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "image_pad_offset_xy": list(offset)}
    layout = TokenLayout(valid, visual, prompt=valid, token_ids=ids)
    values = {key: value.to(device=device, dtype=dtype if value.is_floating_point() else None)
              if isinstance(value, torch.Tensor) else value for key, value in values.items()}
    masks = {name: mask.to(device) for name, mask in masks.items()}
    return ProbeInputs(values, layout), masks, info

def preprocess(image: Image.Image) -> tuple[Image.Image, dict]:
    """Match torchvision's PIL Resize(256), CenterCrop(224) author setup."""
    image = image.convert("RGB")
    width, height = image.size
    resized = ((256, int(256 * height / width)) if width <= height
               else (int(256 * width / height), 256))
    left, top = round((resized[0] - 224) / 2), round((resized[1] - 224) / 2)
    crop = (left, top, left + 224, top + 224)
    return image.resize(resized, Image.Resampling.BILINEAR).crop(crop), {
        "original_size": [width, height], "resize_size": list(resized),
        "crop_in_resized_image": list(crop), "model_input_size": [224, 224],
    }

def delete_patches(pixels, indices, *, patch_size: int):
    """Replace exact non-overlapping patch IDs by zero normalized pixels."""
    import torch
    if pixels.ndim != 4 or len(pixels) != 1 or pixels.shape[-1] % patch_size or pixels.shape[-2] % patch_size:
        raise ValueError("expected one image with a complete non-overlapping patch grid")
    _, channels, height, width = pixels.shape
    rows, columns = height // patch_size, width // patch_size
    edited = pixels.clone().reshape(1, channels, rows, patch_size, columns, patch_size)
    edited = edited.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(rows * columns, channels, patch_size, patch_size)
    selected = torch.as_tensor(indices, device=pixels.device, dtype=torch.long)
    if selected.numel() and ((selected < 0).any() or (selected >= rows * columns).any()):
        raise ValueError("patch index outside the image grid")
    edited[selected] = 0
    return edited.reshape(1, rows, columns, channels, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape_as(pixels)

def add_prefix(inputs, token_ids, *, probe=None):
    """Append fixed assistant-prefix IDs without retokenizing generated text."""
    values = dict(inputs)
    suffix = torch.as_tensor(token_ids, dtype=torch.long, device=values["input_ids"].device)[None]
    values["input_ids"] = torch.cat((values["input_ids"], suffix), dim=1)
    if "attention_mask" in values:
        values["attention_mask"] = torch.cat((values["attention_mask"], torch.ones_like(suffix)), dim=1)
    # Position-dependent fields must be regenerated for the current full prefix.
    values.pop("position_ids", None)
    if probe is None:
        return values
    original = probe.spec.layout(inputs)
    valid = torch.cat((original.valid, torch.ones_like(suffix, dtype=torch.bool)), dim=1)
    visual = (torch.cat((original.visual, torch.zeros_like(suffix, dtype=torch.bool)), dim=1)
              if original.visual is not None else None)
    # Keep the original context boundary: generated text must not become prompt
    # when the same teacher-forced prefix is used by both intervention states.
    prompt = original.valid if original.prompt is None else original.prompt
    prompt = torch.cat((prompt, torch.zeros_like(suffix, dtype=torch.bool)), dim=1)
    expanded_ids = (torch.cat((original.token_ids, suffix), dim=1)
                    if original.token_ids is not None else None)
    return ProbeInputs(values, TokenLayout(valid, visual, prompt, expanded_ids))

def append_answer(inputs, answer_ids):
    """Preserve exact IDs and prompt boundary, marking all answer pieces."""
    values = dict(inputs.kwargs)
    layout = inputs.layout.checked(tuple(values["input_ids"].shape), values["input_ids"].device)
    suffix = torch.tensor(answer_ids, device=values["input_ids"].device, dtype=torch.long)[None]
    values["input_ids"] = torch.cat([values["input_ids"], suffix], 1)
    values["attention_mask"] = torch.cat([values["attention_mask"], torch.ones_like(suffix)], 1)
    values.pop("position_ids", None)
    valid = torch.cat([layout.valid, torch.ones_like(suffix, dtype=torch.bool)], 1)
    visual = torch.cat([layout.visual, torch.zeros_like(suffix, dtype=torch.bool)], 1)
    prompt = torch.cat([layout.valid, torch.zeros_like(suffix, dtype=torch.bool)], 1)
    ids = torch.cat([layout.token_ids, suffix], 1)
    answer_mask = valid & ~prompt
    query = torch.zeros_like(valid)
    start = layout.valid.shape[1] - 1
    query[:, start:start + len(answer_ids)] = True
    return ProbeInputs(values, TokenLayout(valid, visual, prompt, ids)), answer_mask, query
