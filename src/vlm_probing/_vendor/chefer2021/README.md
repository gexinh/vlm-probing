# Pinned Chefer CVPR 2021 ViT relevance implementation

Source: [hila-chefer/Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability),
commit [`c3e578f76b954e8528afeaaee26de3f07e3fe559`](https://github.com/hila-chefer/Transformer-Explainability/tree/c3e578f76b954e8528afeaaee26de3f07e3fe559).
The source is MIT licensed; the complete license is retained in [LICENSE](LICENSE).

| Local file | Upstream file | Changes |
| --- | --- | --- |
| `layers.py` | `modules/layers_ours.py` | Byte-identical. |
| `vit.py` | `baselines/ViT/ViT_LRP.py` | Relative imports; equivalent native PyTorch reshapes replace `einops.rearrange`; remove model URL tables, pretrained factories and conversion helpers. All forward and relevance rules remain intact. |
| `weight_init.py` | `baselines/ViT/weight_init.py` | Byte-identical. |
| `layer_helpers.py` | `baselines/ViT/layer_helpers.py` | Byte-identical. |

The public wrapper does not implicitly download weights. It can copy a native
Hugging Face ViT classifier into this architecture with strict state-dict loading.
The wrapper evaluates one image per relevance propagation, replaces hard-coded
CUDA class seeds with device-aware tensors, leaves parameter gradient buffers
unchanged, and restores the caller's training flags.

The separate upstream `ViT_orig_LRP.py` / `layers_lrp.py` implementation is not
included. Do not describe the optional `full` or `last_layer` modes of this
Chefer-rule model as that separate conventional LRP baseline.
