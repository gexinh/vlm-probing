"""Small pretrained image-path validation; accepts an existing HF model/path.

Example: python examples/pretrained_smoke.py /path/to/Qwen3.5-4B --device cuda:0
No generation benchmark or lens calibration is performed.
"""
import argparse
import json
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from vlm_probing import Prober, TokenMargin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    start = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": args.device},
        attn_implementation="eager",
    ).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model)
    p = Prober(model, processor)
    image = Image.new("RGB", (96, 96), "white")
    ImageDraw.Draw(image).rectangle((16, 16, 80, 80), fill="red")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "What color is the square? Answer with one word."},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    processor_options = {"max_pixels": 256 * 256} if model.config.model_type.startswith("qwen") else {}
    inputs = p.prepare(prompt=prompt, image=image, device=args.device, **processor_options)
    last = max(p.spec.residuals)
    result = p.lens.logit(layers=[0, last // 2, last]).run(inputs)
    with torch.no_grad():
        logits = model(**inputs.kwargs, use_cache=False).logits
    positions = result.tensors["positions"]
    expected = logits[positions[:, 0], positions[:, 1]].cpu()
    error = (result.tensors["logits"][-1].float() - expected.float()).abs().max().item()
    torch.testing.assert_close(result.tensors["logits"][-1], expected, atol=0.25, rtol=0.02)
    embed = p.lens.embed().run(inputs, top_k=3)
    attention_layer = min(p.spec.attentions)
    profile = p.attention.profile(layers=[attention_layer], queries="last_prompt").run(inputs)
    ids = [processor.tokenizer.encode(word, add_special_tokens=False) for word in (" red", " blue")]
    if any(len(item) != 1 for item in ids):
        raise ValueError("smoke-test labels must each encode as one token")
    metric = TokenMargin(ids[0][0], ids[1][0])
    noop = p.causal.patch(layers=[0]).run(inputs, source=inputs, metric=metric)
    torch.testing.assert_close(noop.tensors["effect"], torch.zeros_like(noop.tensors["effect"]))
    # Idefics3 removes all-zero tiles as padding; use an inverted image control.
    pixels = inputs.kwargs["pixel_values"]
    corrupt_pixels = -pixels if model.config.model_type == "idefics3" else torch.zeros_like(pixels)
    corrupt = {**inputs.kwargs, "pixel_values": corrupt_pixels}
    patch = p.causal.patch(layers=[0]).run(corrupt, source=inputs, metric=metric)
    report = {
        "model_class": type(model).__name__, "checkpoint": args.model,
        "torch": torch.__version__, "dtype": str(model.dtype),
        "sequence_shape": list(inputs.kwargs["input_ids"].shape),
        "visual_tokens": len(embed.tensors["positions"]),
        "residual_layers": len(p.spec.residuals), "attention_layers": list(p.spec.attentions),
        "logit_lens_max_abs_error": error,
        "self_patch_effect": noop.tensors["effect"].tolist(),
        "image_patch_effect": patch.tensors["effect"].tolist(),
        "profile_finite": bool(torch.isfinite(profile.tensors["group_mass"]).all()),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "gpu_peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 3),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
