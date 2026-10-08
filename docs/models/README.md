# Model Support

[Home](../../README.md) / [Documentation](../README.md)

Automatic adapters target **Transformers 5.3.x**, tested with **5.3.0**. The eight
VLM families below use native Transformers classes. Each guide provides a full
loading and image-input example. [Llama](llama.md) and [GPT-2](gpt2.md) provide text controls;
[ViT](vit.md) provides a pure-vision classification control.

| Family | Example size | Validation in this release |
| --- | --- | --- |
| [Qwen3.5](qwen3_5.md) | 4B | Tiny architecture + eight pretrained report cases |
| [Qwen3-VL](qwen3_vl.md) | 2B | Tiny architecture + eight pretrained report cases |
| [Qwen2.5-VL](qwen2_5_vl.md) | 3B | Tiny architecture + local pretrained checkpoint |
| [Qwen2-VL](qwen2_vl.md) | 2B | Tiny architecture with actual vision path |
| [SmolVLM](smolvlm.md) | 256M / 500M | Tiny architecture + pretrained cases; 500M Tuned calibration |
| [InternVL](internvl.md) | 1B | Tiny architecture + eight pretrained report cases |
| [LLaVA](llava.md) | 7B | Tiny architecture + pretrained GQA/COCO demos and shared report cases |
| [Pixtral](pixtral.md) | 12B | Native Mistral/Pixtral architecture tests + eight pretrained cases; seven attention methods |
| [ViT](vit.md) | ViT-B/16 | Tiny classifier + actual pretrained author-checkpoint map/deletion controls |

Tiny tests instantiate real vision towers, projectors, and decoders with random
weights. They check all available lens methods, gradients, padding, visual token
coordinates, readout equality, head projection, self-patching, and nontrivial
interventions. This is architecture validation; it does not establish pretrained
accuracy on the checkpoints whose weights were not run. The
[technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf) adds actual execution on five
pretrained VLMs, eight fixed cases each, with explicit capability boundaries.
All eight official processors were also exercised on image+chat inputs.
[Architecture checks](../../examples/hf_model_matrix.py) use small native models.

## Capability matrix

| Methods | Qwen3.5-4B | Other seven VLM families |
| --- | --- | --- |
| Logit, Tuned, Jacobian, Embed, Patchscope | All residual/embedding sites | All residual/embedding sites |
| Attention Lens | Standard-attention layers | All language attention layers |
| Attention profile, head logit attribution | Standard-attention layers | All language attention layers |
| Attention rollout, relevance | Unavailable across the hybrid stack | Consecutive language attention layers from 0 |
| Activation patching, attribution patching, steering | All residual layers | All residual layers |
| IOI-style path patching | Unavailable: linear-attention outputs cannot all be frozen | Head → Q/K/V or final residual, every language attention layer |
| Attention knockout, temperature | Full-attention layers only | Native eager Llama, Mistral, Qwen2, Qwen2-VL, Qwen2.5-VL and Qwen3-VL score taps |
| Attention probability reweighting / ATTATTR | Full-attention layers only | Consumed editable probabilities in the same audited eager decoders |
| EAP-IG | Requires explicit edge sites | Requires explicit edge sites; original input-path graph available for GPT-2 text |

Availability describes the adapter contract. Tuned, Attention, and Jacobian
Lens still require checkpoint-specific calibration or a compatible artifact;
the table does not imply that fitted decoders are supplied for every model.
Grad-CAM can inspect individual full-attention layers, including Qwen3.5.
TAM and Beyond Intuition require the same consecutive attention stack as
Rollout, so the hybrid Qwen3.5 stack is outside their current scope.

Attention profiles, rollout, and relevance require `attn_implementation="eager"`.
Logit/residual probes and projected-head lenses also work with SDPA. Fused
backends do not expose probability matrices. `probe.describe()` reports the
actual sites available on the supplied instance. Keep the attention backend
fixed after constructing a Prober; construct a new one after changing it.
Path patching also works with SDPA because it edits Q/K/V projections and head
messages directly. For GQA, K/V receiver indices name physical shared KV heads.
Path controls and GPT-2 were also checked on random-weight native architectures.
The [path notebook](../../demos/path_patching_demo.ipynb) includes measured pretrained examples.

The [ViT vision-control adapter](vit.md) supports class-conditioned Grad-CAM,
ATTATTR, TAM, Beyond Intuition and genuine Chefer DTD/LRP with the same classifier
weights. DTD is scoped to this explicit ViT backend. VLM use of the other
attribution maps is a documented method transfer, not evidence of full VLM DTD.

## Input and execution contract

Pass the complete processor output, including grid and mask metadata. Probing
uses uncached, full-sequence forwards. All positions refer to the expanded
language sequence, including visual slots and padding. Multiple images within a
prompt, video, quantization, sharded/offloaded models, and compiled models require
additional validation; this release tests unquantized image+text models on CPU
and single-GPU bf16 execution for pretrained VLMs. GPT-2 EAP-IG uses float32.

A residual site is measured after a language block and any intervening visual
injection, before the next block or final readout norm. Image tokens are selected
using explicit architecture token IDs. No fixed-length visual-token guess is
used. For custom layouts and architectures, see [adapters](../ADAPTERS.md).

Legacy Transformers 4.57.x remains supported for Llama, Qwen2.5-VL, and GPT-2.
The expanded model set uses the 5.3.x extra.
