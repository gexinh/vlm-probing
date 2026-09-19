# LLaVA

[Home](../../README.md) / [Models](README.md) / LLaVA

Example checkpoint: [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
Native class: `LlavaForConditionalGeneration`. Install `pip install -e ".[transformers]"` from the repository root.

## Load and probe

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlm_probing import Prober

model_id = "llava-hf/llava-1.5-7b-hf"
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

result = probe.lens.logit(layers=[0, 15, 31]).run(inputs)
visual = probe.lens.embed().run(inputs, top_k=5)
profile = probe.attention.profile(layers=[0], queries="last_prompt").run(inputs)
print(result.tensors["positions"], result.tensors["logits"].shape)
```

Layer choices above apply to the example checkpoint. Use `describe()` when
loading a different size. Pass native processor dictionaries directly to
`.run(...)` if you already have prepared inputs. On CPU, use float32 and a
small checkpoint/configuration instead of the CUDA/bf16 settings above.

## Adapter behavior

Supports the native `LlavaForConditionalGeneration` class with a Llama text backbone. The processor must expand each image placeholder to the actual visual token sequence. Legacy unexpanded input layouts are not inferred or silently remapped.

All six lenses, four observational attention methods, and four residual causal methods are available with eager attention. This is the classic LLaVA family; LLaVA-NeXT, OneVision, and original remote-code wrappers have different contracts and are not registered under this adapter.

Native HF attention outputs are observational. Knockout, probability reweighting,
attention temperature, and EAP-IG require the explicit editable sites described
in the [capability matrix](README.md#capability-matrix).

## Validation

Validated with a tiny randomly initialized instance of this native architecture, including real image pixels, the vision tower, projector, and decoder. The named pretrained checkpoint is a loading example; its weights were not run in this release.

See [test coverage and results](../releases/v0.3.0.md), the
[executable matrix](../../examples/hf_model_matrix.py), and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/llava/modeling_llava.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
