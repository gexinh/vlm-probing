# ViT vision-classification control

[Home](../../README.md) / [Models](README.md)

Native `transformers.ViTForImageClassification` is supported on Transformers
5.3.x with eager attention. This is a pure-vision control for spatial
attribution comparisons; it has no language vocabulary lens.

```python
from transformers import AutoImageProcessor, ViTForImageClassification
from vlm_probing import ClassScore, Prober

# Supply a locally prepared checkpoint and an actual PIL image.
checkpoint = "/path/to/downloaded/vit"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = ViTForImageClassification.from_pretrained(
    checkpoint, attn_implementation="eager",
).eval()
inputs = dict(processor(images=image, return_tensors="pt"))
probe = Prober(model)
target = 340  # Replace with a verified model.config.id2label class index.
metric = ClassScore(target)

cam = probe.attention.grad_cam(layers=[11]).run(inputs, metric=metric)
ig = probe.attention.attribution(layers=[11], steps=20).run(inputs, metric=metric)
tam = probe.attention.tam(steps=20).run(inputs, metric=metric)
bt = probe.attention.beyond_intuition(variant="token", steps=20).run(inputs, metric=metric)
# Actual author relevance rules on a copy of precisely these classifier weights.
dtd = probe.attention.dtd_lrp().run(inputs["pixel_values"], target=target)
print(dtd.tensors["patch_relevance"].shape)
```

Map factories default to CLS query `[0]` and spatial keys `"visual"` for this
adapter. Native logits remain `[B,C]`; `ClassScore(kind="logit")` names the
selected class explicitly. The adapter exposes consumed probabilities and
masked scores without changing weights; no-op forwards are bit-for-bit equal
in the CPU check. Patch coordinates are row-major after CLS and refer to the
processor's fixed resized/cropped image, not automatically to the source image.
Interpolated position grids and distilled DeiT variants are not declared native
support in this adapter.

DTD requires the eligible author architecture; unsupported activation/config
variants raise an error rather than substituting a gradient recurrence. Its
factory creates a separate weight copy, so account for that memory. Both architecture tests and pretrained author-checkpoint demonstrations are
available. The [ViT attention notebook](../../demos/attention_comparison_demo.ipynb)
compares maps and measured patch-deletion effects on author sample images.
