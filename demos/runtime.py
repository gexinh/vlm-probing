"""Optional real model runs. No model or fitted weights ship with the demos."""
import json
import os
from pathlib import Path

import torch
from PIL import Image

from vlm_probing import Prober
from demos.inputs import prepare_case, append_answer
from demos.path_helpers import load_model


def bind_model(default_id, *, attention=True):
    """Select a native checkpoint/device using environment variables."""
    model_id = os.environ.get("VLM_PROBING_MODEL", default_id)
    device = os.environ.get("VLM_PROBING_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_id, device=device,
                                  attn_implementation="eager" if attention else "sdpa")
    return Prober(model, processor=processor), processor, device


def gqa_case(probe, processor, sample, image_directory, *, score_answer=True):
    """Paper-aligned LLaVA preprocessing with explicit visual/question masks.

    Replace this dataset adapter when changing architectures or input formats;
    the Prober factories and plotting helpers remain the same. In particular,
    this recipe requires a LLaVA processor with expanded image placeholders.
    """
    parameter = next(probe.model.parameters())
    base, masks, info = prepare_case(sample, image_directory, processor,
                                     parameter.device, dtype=parameter.dtype)
    if not score_answer:
        return base, masks, info
    target = sample["answer"].capitalize()
    ids = processor.tokenizer.encode(target, add_special_tokens=False)
    scored, answer_mask, prediction_mask = append_answer(base, ids)
    return base, scored, answer_mask, prediction_mask, masks, info


def prepared_chat(probe, processor, prompt, image=None):
    """Prepare native HF inputs for a single chat; never retokenize hidden IDs."""
    parameter = next(probe.model.parameters())
    if getattr(probe.model.config, "model_type", None) == "llava":
        from demos.inputs import pad_square
        text = f"USER: {'<image> ' if image is not None else ''}{prompt} ASSISTANT:"
        if image is not None:
            image, _ = pad_square(image, processor.image_processor.image_mean)
            values = dict(processor(text=text, images=image, return_tensors="pt"))
        else:
            values = dict(processor.tokenizer(text, return_tensors="pt"))
    else:
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": prompt}]
        text = processor.apply_chat_template([{ "role": "user", "content": content }],
                                              tokenize=False, add_generation_prompt=True)
        values = dict(processor(text=[text], **({"images": [image]} if image is not None else {}),
                                return_tensors="pt"))
    return {key: value.to(device=parameter.device,
                         dtype=parameter.dtype if value.is_floating_point() else value.dtype)
            if isinstance(value, torch.Tensor) else value for key, value in values.items()}


def fitted_method(probe, name, *, tokens="text"):
    """Load caller-supplied per-model fitted artifacts and explicit binding IDs.

    VLM_PROBING_LENS_CONFIG points to JSON with method names mapped to
    {"directory": relative-or-absolute-path, "binding": {...}, "layers": [...]}.
    Relative directories resolve beside that config. Binding must correspond to
    the exact frozen model, tokenizer, readout, and calibration data.
    """
    config_name = os.environ.get("VLM_PROBING_LENS_CONFIG")
    if not config_name:
        raise ValueError("Set VLM_PROBING_LENS_CONFIG to your model-specific fitted-lens configuration")
    config_path = Path(config_name).expanduser().resolve()
    specification = json.loads(config_path.read_text())[name]
    directory = Path(specification["directory"]).expanduser()
    if not directory.is_absolute():
        directory = config_path.parent / directory
    factory = getattr(probe.lens, name)
    return factory(layers=specification["layers"], tokens=tokens,
                   binding=specification["binding"]).load(directory)


def attention_methods(probe, *, queries, keys, input_key="pixel_values", steps=20):
    """Eight public decoder/ViT analyses; inspect capabilities before running."""
    last = max(probe.spec.attentions)
    return {
        "Raw attention": (probe.attention.profile(layers=[last], queries=queries,
                                                   include_attention=True), "attention"),
        "Grad-CAM": (probe.attention.grad_cam(layers=[last], queries=queries, keys=keys), "cam"),
        "ATTATTR": (probe.attention.attribution(layers=[last], queries=queries, keys=keys,
                                               steps=steps, quadrature="left"), "attribution"),
        "Rollout": (probe.attention.rollout(), "rollout"),
        "TAM": (probe.attention.tam(queries=queries, keys=keys, steps=steps,
                                    quadrature="left", input_key=input_key), "map"),
        "Beyond Intuition-H": (probe.attention.beyond_intuition(queries=queries, keys=keys,
                                variant="head", steps=steps, quadrature="left", input_key=input_key), "map"),
        "Beyond Intuition-C": (probe.attention.beyond_intuition(queries=queries, keys=keys,
                                variant="token", steps=steps, quadrature="left", input_key=input_key), "map"),
        "Generic relevance": (probe.attention.relevance(), "relevance"),
    }
