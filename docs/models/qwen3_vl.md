# Qwen3-VL

[Home](../../README.md) / [Models](README.md) / Qwen3-VL

Example checkpoint: [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
Native class: `Qwen3VLForConditionalGeneration`. Install `pip install -e ".[transformers]"` from the repository root.

## Load and probe

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlm_probing import Prober

model_id = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id)
probe = Prober(model, processor)

messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)
inputs = probe.prepare(
    prompt=prompt, image=Image.open("example.jpg").convert("RGB"), device="cuda:0",
)
print(probe.describe())

result = probe.lens.logit(layers=[0, 13, 27]).run(inputs)
visual = probe.lens.embed().run(inputs, top_k=5)
profile = probe.attention.profile(layers=[0], queries="last_prompt").run(inputs)
print(result.tensors["positions"], result.tensors["logits"].shape)
```

Layer choices above apply to the example checkpoint. Use `describe()` when
loading a different size. Pass native processor dictionaries directly to
`.run(...)` if you already have prepared inputs. On CPU, use float32 and a
small checkpoint/configuration instead of the CUDA/bf16 settings above.

## Adapter behavior

Residual layer `i` is captured after its DeepStack visual addition, at the next decoder block's input or the final norm's input. Capturing the decoder block output alone would omit that addition.

Profiles and head lenses inspect language self-attention. Rollout/relevance summarize those attention paths; they do not model the extra DeepStack injection edges. Image grids are included in clean/corrupt alignment checks.

Native HF attention outputs are observational. Knockout, probability reweighting,
attention temperature, and EAP-IG require the explicit editable sites described
in the [capability matrix](README.md#capability-matrix).

## Validation

Validated with a tiny randomly initialized instance of this native architecture, including real image pixels, the vision tower, projector, and decoder. The named pretrained checkpoint is a loading example; its weights were not run in this release.

See [test coverage and results](../releases/v0.3.0.md), the
[executable matrix](../../examples/hf_model_matrix.py), and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
