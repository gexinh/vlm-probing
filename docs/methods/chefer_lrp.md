# Chefer Transformer Attribution (DTD relevance)

[Home](../../README.md) / [Methods](README.md) / Chefer Transformer Attribution

This is the proposed method from [Transformer Interpretability Beyond Attention
Visualization, CVPR 2021](https://arxiv.org/abs/2012.09838). It uses the pinned
[official ViT implementation](https://github.com/hila-chefer/Transformer-Explainability/tree/c3e578f76b954e8528afeaaee26de3f07e3fe559),
including relevance propagation through linear layers, attention matrix products
and residual branches. A forward attention matrix alone is insufficient.

The [executed ViT comparison](../../demos/attention_comparison_demo.ipynb)
uses the author checkpoint and two official images, checks native/DTD logit
parity, and displays attribution maps alongside actual patch-deletion effects.

## Usage with a native ViT classifier

For an existing native ViT `probe`, the common facade is
`probe.attention.dtd_lrp().run(inputs["pixel_values"], target=class_id)`.
The explicit form below also works without constructing a Prober.

```python
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from vlm_probing.attention.lrp import CheferLRP, create_chefer_vit

# Reuse an already-loaded classifier. No second weight download is needed.
classifier = ViTForImageClassification.from_pretrained(local_model_directory)
processor = ViTImageProcessor.from_pretrained(local_model_directory)
pixels = processor(images=image, return_tensors="pt")["pixel_values"]
backend = create_chefer_vit(source=classifier)
result = CheferLRP(backend).run(pixels, target=class_id)
heatmap = result.tensors["patch_relevance"]  # [B, grid_height, grid_width]
```

The native Hugging Face classifier must use exact GELU, a linear classification
head, and the ViT architecture. Input image sizes must match its configured patch
grid. LayerNorm epsilon is copied explicitly. Image processing is caller-owned.
The result includes the selected class, classifier logits, LRP attention CAMs,
attention gradients and patch relevance. Upsample patch relevance only for
display; it is not a pixel-level segmentation prediction.

## Method

After the classification forward, use a one-hot relevance seed for class `c` and
propagate it backward with the official rules (`alpha=1`). This produces
attention relevance `R_A`, which differs from forward probabilities `A`.
Backpropagate the selected class logit through the same forward to obtain
`dScore/dA`. Aggregate:

```text
C_l = mean_heads(relu(R_A,l * dScore/dA_l))
R = I
R <- (I + C_l) @ R                 # ascending layers; no row normalization
map = R[CLS, image_patch_tokens]
```

The public tensor kernel is
`CheferTransformerAttribution(start_layer=0).run(attention_relevance, gradients)`.
Both inputs have shape `[layer,batch,head,query,key]`. This kernel aggregates
already-computed CAMs; it cannot compute DTD relevance from probabilities alone.

## API and scope

```text
create_chefer_vit(*, source=None, checkpoint=None, **configuration)
CheferLRP(model, *, method="transformer_attribution", start_layer=0, result_device="cpu")
method.run(pixel_values, *, target=None)
```

`target=None` selects each image's top class. An integer selects one class for all
images; a long `[B]` tensor selects a class per image. Batched inputs are processed
as independent single-image relevance runs, matching the author implementation.
`start_layer` controls aggregation. The original evaluation scripts often use
`start_layer=1`; it must be recorded when comparing results.

`method="transformer_attribution"` is the CVPR proposed method, often labelled
**Chefer** or **DTD-LRP** by later comparisons. Optional `full` returns pixel
relevance using these same Chefer rules, and `last_layer` returns their last-layer
CAM. These optional modes are **not** the separate conventional LRP and partial
LRP baselines in the CVPR figure, which use `ViT_orig_LRP.py`.

This backend supports vision classification. It is not registered as a general
decoder probe: existing LLM/VLM adapters do not provide all model-specific DTD
rules. The existing [`attention.relevance`](attention_relevance.md) method instead
implements the self-attention recurrence from the **ICCV 2021 Generic
Attention-model Explainability** paper, which deliberately removes LRP.

[Model capability matrix](../models/README.md) ·
[Wrapper](../../src/vlm_probing/attention/lrp.py) ·
[Aggregation kernel](../../src/vlm_probing/attention/chefer.py) ·
[Source provenance](../../src/vlm_probing/_vendor/chefer2021/README.md)
