# Qwen3.5

[Home](../../README.md) / [Models](README.md) / Qwen3.5

Example checkpoint: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
Native class: `Qwen3_5ForConditionalGeneration`. Install `pip install -e ".[transformers]"` from the repository root.

## Load and probe

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlm_probing import Prober

model_id = "Qwen/Qwen3.5-4B"
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
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = probe.prepare(
    prompt=prompt, image=Image.open("example.jpg").convert("RGB"), device="cuda:0",
)
print(probe.describe())

result = probe.lens.logit(layers=[7, 15, 31]).run(inputs)
visual = probe.lens.embed().run(inputs, top_k=5)
profile = probe.attention.profile(layers=[3], queries="last_prompt").run(inputs)
print(result.tensors["positions"], result.tensors["logits"].shape)
```

Layer choices above apply to the example checkpoint. Use `describe()` when
loading a different size. Pass native processor dictionaries directly to
`.run(...)` if you already have prepared inputs. On CPU, use float32 and a
small checkpoint/configuration instead of the CUDA/bf16 settings above.

## Adapter behavior

Qwen3.5-4B has 32 language layers. Residual lenses and interventions cover all 32. Attention Lens, attention profiles, and head logit attribution cover full-attention layers **3, 7, 11, 15, 19, 23, 27, 31**. Linear-attention layers do not expose a token-by-token softmax matrix, so rollout and relevance are unavailable on this checkpoint.

The adapter captures gated values immediately before the attention output projection and handles the final RMSNorm's `1 + weight` gain. `H * head_dim` need not equal the residual width. Transformers' PyTorch fallback works without the optional fast linear-attention kernels.

Native HF attention outputs are observational. Knockout, probability reweighting,
attention temperature, and EAP-IG require the explicit editable sites described
in the [capability matrix](README.md#capability-matrix).

## Validation

Both the tiny architecture and the named local pretrained checkpoint were run. Pretrained checks cover an image prompt, selected Logit Lens layers, EmbedLens, an attention profile, self-patching, and an image intervention.

See [test coverage and results](../releases/v0.3.0.md), the
[executable matrix](../../examples/hf_model_matrix.py), and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
