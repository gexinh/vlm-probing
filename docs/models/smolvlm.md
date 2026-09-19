# SmolVLM

[Home](../../README.md) / [Models](README.md) / SmolVLM

Example checkpoint: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).
Native class: `Idefics3ForConditionalGeneration`. Install `pip install -e ".[transformers]"` from the repository root.

## Load and probe

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlm_probing import Prober

model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
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

result = probe.lens.logit(layers=[0]).run(inputs)
visual = probe.lens.embed().run(inputs, top_k=5)
profile = probe.attention.profile(layers=[0], queries="last_prompt").run(inputs)
print(result.tensors["positions"], result.tensors["logits"].shape)
```

Layer choices above apply to the example checkpoint. Use `describe()` when
loading a different size. Pass native processor dictionaries directly to
`.run(...)` if you already have prepared inputs. On CPU, use float32 and a
small checkpoint/configuration instead of the CUDA/bf16 settings above.

## Adapter behavior

SmolVLM-256M-Instruct is registered by its actual checkpoint class, `Idefics3ForConditionalGeneration`, with a Llama text backbone at `model.text_model`. The processor handles image crops, visual placeholders, and pixel masks; pass all returned fields to the model. Layouts mark only the image-content slots, excluding image boundary tokens. SmolVLM2 checkpoints using the separate `SmolVLMForConditionalGeneration` class are not part of this adapter.

SmolVLM treats an all-zero image tile as padding. For a blank-image control, process a real blank PIL image through the same processor rather than zeroing `pixel_values`. All six lenses, four observational attention methods, and four residual causal methods are available with eager attention.

Native HF attention outputs are observational. Knockout, probability reweighting,
attention temperature, and EAP-IG require the explicit editable sites described
in the [capability matrix](README.md#capability-matrix).

## Validation

Both the tiny architecture and the named pretrained checkpoint were run. Pretrained checks cover an image prompt, selected Logit Lens layers, EmbedLens, an attention profile, self-patching, and an image intervention. The processor expanded the image to 1,088 visual slots, all correctly identified by the adapter.

See [test coverage and results](../releases/v0.3.0.md), the
[executable matrix](../../examples/hf_model_matrix.py), and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/idefics3/modeling_idefics3.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
