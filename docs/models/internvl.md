# InternVL

[Home](../../README.md) / [Models](README.md) / InternVL

Example checkpoint: [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf).
Native class: `InternVLForConditionalGeneration`. Install `pip install -e ".[transformers]"` from the repository root.

## Load and probe

```python
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from vlm_probing import Prober

model_id = "OpenGVLab/InternVL3-1B-hf"
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

result = probe.lens.logit(layers=[0, 11, 23]).run(inputs)
visual = probe.lens.embed().run(inputs, top_k=5)
profile = probe.attention.profile(layers=[0], queries="last_prompt").run(inputs)
print(result.tensors["positions"], result.tensors["logits"].shape)
```

Layer choices above apply to the example checkpoint. Use `describe()` when
loading a different size. Pass native processor dictionaries directly to
`.run(...)` if you already have prepared inputs. On CPU, use float32 and a
small checkpoint/configuration instead of the CUDA/bf16 settings above.

## Adapter behavior

Use the **HF-native `-hf` checkpoint**, whose class is `InternVLForConditionalGeneration`. The automatic adapter accepts its audited Qwen2 or Llama text backbone. The 1B example uses Qwen2. Original remote-code InternVL wrappers require a separate explicit adapter.

The official processor handles dynamic tiles and thumbnail tokens. All six lenses, four observational attention methods, and four residual causal methods are available with eager attention.

The eager adapter exposes editable pre-softmax scores and consumed attention
probabilities. Knockout, temperature, reweighting, and ATTATTR operate on actual
decoder computation. Original EAP-IG remains scoped to the audited GPT-2 graph.

## Validation

Validated with a tiny randomly initialized instance including the real vision
path, and with the named pretrained checkpoint on eight fixed GQA/MMVP/VSR/
VisOnlyQA report cases. Readouts, attention attribution, real residual edits
and one complete paired-image path sweep are recorded in the
[technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf).

See the [architecture checks](../../examples/hf_model_matrix.py) and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/internvl/modeling_internvl.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
