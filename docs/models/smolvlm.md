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

SmolVLM treats an all-zero image tile as padding. For a blank-image control, process a real blank PIL image through the same processor rather than zeroing `pixel_values`. The same constraint applies to input-path integration: endpoint-inclusive sampling needs a baseline that preserves the valid tile layout. Lens readouts, observational attention maps, and residual causal methods are available with eager attention.

Its native Llama backbone exposes audited score and consumed-probability taps
with Transformers 5.3 eager attention. These enable knockout, temperature,
reweighting and attention-space attribution. EAP-IG requires an explicit
validated edge graph. See the
[capability matrix](README.md#capability-matrix).

## Validation

SmolVLM-500M-Instruct also runs all eight fixed report cases with actual
residual replacements, editable attention and a full paired-image path sweep.
Three Tuned Lens translators are independently calibrated on COCO captions.
See the [technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf) and
[VLM lens notebook](../../demos/vlm_lens_demo.ipynb).

Both the tiny architecture and the named pretrained checkpoint were run. Pretrained checks cover an image prompt, selected Logit Lens layers, EmbedLens, an attention profile, self-patching, and an image intervention. The processor expanded the image to 1,088 visual slots, all correctly identified by the adapter.

See the [architecture checks](../../examples/hf_model_matrix.py) and the
[official model implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/idefics3/modeling_idefics3.py).

Next: [method guides](../methods/README.md) and [metrics/calibration](../API.md).
