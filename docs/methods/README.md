# Method Guides

[Home](../../README.md) / [Documentation](../README.md)

Each guide explains the computation, includes a usage example, and documents
parameters and returned tensors. Availability depends on the model adapter;
check `probe.describe()` or the [model matrix](../models/README.md).

## Lens

| Method | Public factory | What it does |
| --- | --- | --- |
| [Logit Lens](logit_lens.md) | `probe.lens.logit(...)` | Decode intermediate residuals with the model's final normalization and vocabulary head. |
| [Tuned Lens](tuned_lens.md) | `probe.lens.tuned(...)` | Fit one affine translator per residual layer before applying the model's frozen readout. |
| [Attention Lens](attention_lens.md) | `probe.lens.attention(...)` | Train separate vocabulary decoders for individual attention heads. |
| [Jacobian Lens](jacobian_lens.md) | `probe.lens.jacobian(...)` | Estimate a layer-to-final-residual Jacobian, transport representations, and decode them. |
| [EmbedLens](embed_lens.md) | `probe.lens.embed(...)` | Find vocabulary embeddings nearest to projected visual tokens. |
| [Patchscope](patchscope.md) | `probe.lens.patchscope(...)` | Insert a source representation into a target prompt and inspect its output logits. |

## Attention

| Method | Public factory | What it does |
| --- | --- | --- |
| [Attention Profile](attention_profile.md) | `probe.attention.profile(...)` | Measure attention entropy, concentration, and mass assigned to token groups. |
| [Head Logit Attribution](head_logit_attribution.md) | `probe.attention.head_logits(...)` | Project head contributions to vocabulary logits with a fixed final normalization scale. |
| [Attention Rollout](attention_rollout.md) | `probe.attention.rollout(...)` | Compose normalized self-attention transitions across consecutive layers. |
| [Attention Relevance](attention_relevance.md) | `probe.attention.relevance(...)` | Propagate positive gradient-weighted self-attention for a selected output score. |
| [Attention Reweighting](attention_reweight.md) | `probe.attention.reweight(...)` | Rescale attention on selected key tokens and measure the output change. |
| [Attention Temperature](attention_temperature.md) | `probe.attention.temperature(...)` | Change the sharpness of attention at selected query rows. |

## Causal

| Method | Public factory | What it does |
| --- | --- | --- |
| [Activation Patching](activation_patching.md) | `probe.causal.patch(...)` | Replace selected receiver residuals with source residuals and measure the causal effect. |
| [Ablation](ablation.md) | `probe.causal.ablate(...)` | Replace selected residual activations with zeros, a reference mean, or a reference sample. |
| [Attribution Patching](attribution_patching.md) | `probe.causal.attribute(...)` | Approximate a patch effect using activation differences and a receiver-side gradient. |
| [Attention Knockout](attention_knockout.md) | `probe.causal.knockout(...)` | Block selected query-to-key attention paths before softmax. |
| [EAP-IG (explicit edge activations)](eap_ig.md) | `probe.causal.eap_ig(...)` | Integrate gradients along a simultaneous interpolation of declared edge messages. |
| [Residual Steering](steering.md) | `probe.causal.steer(...)` | Add a supplied direction to selected residuals and measure the output change. |

## Example setup

For a downloaded VLM, start with its [model guide](../models/README.md) to obtain
`probe` and `inputs`. For a small, fully runnable demonstration of every method,
run this setup from the repository root after installation. Method-page snippets
use these variables. TinyModel exposes editable attention and real edge sites,
so it also demonstrates the four methods unavailable on native HF adapters.

```python
import torch
from examples.tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin

torch.manual_seed(7)
torch.set_num_threads(1)
model = TinyModel().eval()
probe = Prober(model)
inputs = {"input_ids": torch.tensor([[1, 2, 3], [3, 4, 5]]),
          "image_tokens": torch.randn(2, 2, 6)}
corrupt = {**inputs, "image_tokens": torch.zeros_like(inputs["image_tokens"])}
calibration = inputs
evaluation = {**inputs, "image_tokens": torch.randn(2, 2, 6)}
metric = TokenMargin(positive=4, negative=5)
```

Here vocabulary IDs 4 and 5 are synthetic labels. For a pretrained model, choose
verified single-token labels with its tokenizer, or use
[SequenceLogProb](../API.md#metrics-and-explicit-layouts) for full answers.
For causal comparisons, prepare aligned clean/corrupt inputs. For fitted lenses,
use a representative calibration set and separate evaluation data.

## Shared conventions

`layers=None` selects available method sites; indices are zero-based. Tokens may
be `"all"`, `"visual"`, `"text"`, `"last_prompt"`, a position/list, or a boolean
`[B,T]` mask. Positions refer to the expanded language sequence. `"last_prompt"`
means the last valid token unless an explicit prompt mask is supplied.

In output shapes, `L` is the number of selected layers, `B` batch size, `T/Q`
sequence/query length, `H` heads, `V` vocabulary size, and `N` packed selected
positions. Lens `positions[N,2]` maps packed rows to batch/token coordinates.
Layer metadata preserves the chosen order. Results default to detached CPU tensors;
`result.save(directory)` writes tensors and JSON metadata.

See [the full API](../API.md), [custom adapters](../ADAPTERS.md), and the
[paper index](../REFERENCES.md) for shared contracts and sources.
