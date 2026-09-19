# Model Support

[Home](../../README.md) / [Documentation](../README.md)

Automatic adapters target **Transformers 5.3.x**, tested with **5.3.0**. The seven
VLM families below use native Transformers classes. Each guide provides a full
loading and image-input example. Llama is also retained as a [text-only control](llama.md).

| Family | Example size | Validation in this release |
| --- | --- | --- |
| [Qwen3.5](qwen3_5.md) | 4B | Tiny architecture + local pretrained checkpoint |
| [Qwen3-VL](qwen3_vl.md) | 2B | Tiny architecture with actual vision path |
| [Qwen2.5-VL](qwen2_5_vl.md) | 3B | Tiny architecture + local pretrained checkpoint |
| [Qwen2-VL](qwen2_vl.md) | 2B | Tiny architecture with actual vision path |
| [SmolVLM](smolvlm.md) | 256M | Tiny architecture + pretrained checkpoint |
| [InternVL](internvl.md) | 1B | Tiny architecture with actual vision path |
| [LLaVA](llava.md) | 7B | Tiny architecture with actual vision path |

Tiny tests instantiate real vision towers, projectors, and decoders with random
weights. They check all available lens methods, gradients, padding, visual token
coordinates, readout equality, head projection, self-patching, and nontrivial
interventions. This is architecture validation; it does not establish pretrained
accuracy on the four checkpoints whose weights were not run. All seven official
processors were also exercised on image+chat inputs. See the
[validation report](../releases/v0.3.0.md).

## Capability matrix

| Methods | Qwen3.5-4B | Other six VLM families |
| --- | --- | --- |
| Logit, Tuned, Jacobian, Embed, Patchscope | All residual/embedding sites | All residual/embedding sites |
| Attention Lens | Standard-attention layers | All language attention layers |
| Attention profile, head logit attribution | Standard-attention layers | All language attention layers |
| Attention rollout, relevance | Unavailable across the hybrid stack | Consecutive language attention layers from 0 |
| Activation patching, ablation, attribution patching, steering | All residual layers | All residual layers |
| Attention knockout, reweighting, temperature | Requires custom intervention sites | Requires custom intervention sites |
| EAP-IG | Requires explicit edge sites | Requires explicit edge sites |

Attention profiles, rollout, and relevance require `attn_implementation="eager"`.
Logit/residual probes and projected-head lenses also work with SDPA. Fused
backends do not expose probability matrices. `probe.describe()` reports the
actual sites available on the supplied instance. Keep the attention backend
fixed after constructing a Prober; construct a new one after changing it.

## Input and execution contract

Pass the complete processor output, including grid and mask metadata. Probing
uses uncached, full-sequence forwards. All positions refer to the expanded
language sequence, including visual slots and padding. Multiple images within a
prompt, video, quantization, sharded/offloaded models, and compiled models require
additional validation; this release tests unquantized image+text models on CPU
and single-GPU bf16 execution for the three pretrained checkpoints.

A residual site is measured after a language block and any intervening visual
injection, before the next block or final readout norm. Image tokens are selected
using explicit architecture token IDs. No fixed-length visual-token guess is
used. For custom layouts and architectures, see [adapters](../ADAPTERS.md).

Legacy Transformers 4.57.x remains supported for the original Llama and
Qwen2.5-VL adapters. The expanded model set uses the 5.3.x extra.
