# IOI Path Patching

[Home](../../README.md) / [Methods](README.md) / IOI Path Patching

Measure the effect of a sender attention head through selected receiver Q/K/V
activations, or through the final residual stream. This is an actual forward
intervention with the path controls described in Wang et al.'s IOI paper.

## Computation

Let \(x\) be the base input and \(x'\) the donor input. Cache their head outputs
\(z^{\mathrm{base}}_{\ell,h,t}\) and \(z^{\mathrm{donor}}_{\ell,h,t}\). In a
controlled run on \(x\), use

\[
\widetilde z_{\ell,h,t} =
\begin{cases}
z^{\mathrm{donor}}_{\ell,h,\pi(t)}, & (\ell,h)\in S,\ t\in T_s,\\
z^{\mathrm{base}}_{\ell,h,t}, & \text{otherwise},
\end{cases}
\]

where \(S\) selects sender heads and \(\pi\) maps selected base positions to
selected donor positions. Every attention-head output is controlled; MLPs and
LayerNorms recompute normally. Cache the selected receiver activations
\(\widetilde r_R\) from this run. Then run \(x\) again and inject only
\(\widetilde r_R\) at the receiver positions. Report
\(\Delta m=m(x; r_R\leftarrow\widetilde r_R)-m(x)\).

The receiver's Q/K/V is captured before its head output is frozen. Its own frozen
output therefore does not prevent capturing a changed receiver input. The final
run releases all sender and head-freezing controls.

Four forward passes are required: base, donor, controlled base, and receiver-only
base. Multiple senders and receivers configured together form one joint
experiment with `.run()`. `.sweep()` tests each sender independently, shares the
base/donor caches, and executes `2 + 2N` actual forwards for `N` paths.

Freezing all head outputs blocks new attention-mediated propagation between
positions in the controlled run. For a Q receiver, select the same positions as
the sender, such as `"last_prompt"`. To study visual information reaching a later
text prediction, replace a visual sender and capture K/V at those visual
positions; the receiver-only run can then attend to the changed keys or values.

## Example

Run from the repository root after installation. This example uses a small CPU
model with real attention and a caller-supplied image-token prefix.

```python
import torch
from examples.tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin

torch.manual_seed(7)
torch.set_num_threads(1)
probe = Prober(TinyModel().eval())
base = {"input_ids": torch.tensor([[1, 2, 3]]),
        "image_tokens": torch.randn(1, 2, 6)}
donor = {**base, "image_tokens": torch.zeros_like(base["image_tokens"])}
metric = TokenMargin(positive=4, negative=5)

result = probe.causal.path(
    senders=[(0, 0)],
    receivers=[(1, 0, "v")],
    sender_tokens="visual",
    receiver_tokens="visual",
).run(base, donor=donor, metric=metric)
print(result.tensors["effect"])
result.save("results/visual_value_path")
```

For text contrast pairs that intentionally change token IDs, pass
`alignment="position"` after checking the role positions yourself. A custom
position mapping selects the same number of sender and donor positions in each
example:

```python
text_donor = {**base, "input_ids": torch.tensor([[5, 6, 7]])}
mapped = probe.causal.path(
    senders=[(0, 0)],
    receivers="residual",
    sender_tokens=[2, 4],
    donor_tokens=[3, 4],
    receiver_tokens="last_prompt",
).run(base, donor=text_donor, metric=metric, alignment="position")
```

Positions include the two visual slots in this example. Within each batch row,
selected positions are paired in ascending order. Receiver positions always
refer to the controlled base run; donor position mapping affects senders only.

## API

```text
probe.causal.path(
    *, senders, receivers="residual", sender_tokens="last_prompt",
    donor_tokens=None, receiver_tokens="last_prompt"
)
method.run(base_inputs, *, donor, metric, alignment="strict")
method.sweep(base_inputs, *, donor, metric, alignment="strict", progress=None)
```

| Parameter | Meaning |
| --- | --- |
| `senders` | Nonempty list of `(layer, head)` pairs. Layers and physical query-head indices are zero-based. |
| `receivers` | Nonempty list of `(layer, head, "q" / "k" / "v")` triples, or `"residual"` for the final residual before final normalization. |
| `sender_tokens` | Base positions at which donor sender outputs replace base outputs. |
| `donor_tokens` | Donor positions supplying sender outputs; defaults to `sender_tokens`. Each row must select the same positive count as the corresponding base row. |
| `receiver_tokens` | Base positions at which controlled receiver values are injected in the final run. |
| `base_inputs / donor` | Native model kwargs or `ProbeInputs`, with aligned expanded layouts. |
| `metric` | `TokenMargin`, `SequenceLogProb`, or a callable accepting full logits and returning a scalar or `[B]`. |
| `alignment` | `"strict"` checks token IDs, masks, and declared metadata. `"position"` relaxes token-ID equality while retaining structural checks. |

Token selectors accept `"all"`, `"visual"`, `"text"`, `"last_prompt"`, an
integer/list of expanded positions, or a boolean `[B,T]` mask. Sender, donor, and
receiver selections must be nonempty in every example and cannot select padding.

Returns `ProbeResult` with `baseline_score`, `donor_score`, `intervention_score`,
and `effect = intervention_score - baseline_score`. Each is scalar or `[B]`,
matching the metric. There is no layer-sweep dimension. Metadata records the four
stages, path configuration, selected coordinates, and adopted control rule.

With `.sweep()`, `intervention_score` and `effect` are `[N,B]` (or `[N]` for a
scalar metric); baseline and donor scores retain their original `[B]`/scalar
shape. `senders[N,2]` records the coordinate order. Every sender uses the same
receiver set; receiver endpoints are not swept. The optional progress callback
receives `(completed, total, sender)` after each independent path.

```python
senders = [(layer, head) for layer, point in sorted(probe.spec.path_heads.items())
           for head in range(point.heads)]
grid = probe.causal.path(senders=senders, receivers="residual").sweep(
    base, donor=donor, metric=metric,
)
```

## Model contract and limits

Check `probe.describe()["methods"]["causal.path"]` before an experiment. The
adapter must expose editable head messages at every decoder layer, the requested
receiver Q/K/V sites, and the final residual when requested. Partial attention
coverage cannot enforce the experiment's path controls and is rejected.

Native Hugging Face adapters freeze each head's value aggregation at the input
to the attention output projection. For a fixed linear projection, freezing
these messages also freezes their projected residual contributions, while the
projection bias is applied once. Native receivers use projection outputs before
positional rotation; Qwen3-VL uses its normalized Q/K values before rotation.
Later positional operations recompute at the base positions.

For grouped-query attention, Q and sender indices refer to query heads; K/V
indices refer to **physical KV heads**. Editing one shared KV head affects every
query head in its group. The API does not silently reinterpret a query-head index
as an independent K/V head. Qwen3.5's mixed linear/full-attention decoder does not
currently expose complete freeze sites and therefore does not support this method.

Identity donors and sender paths that cannot reach the chosen receiver are valid
controls and should have zero effect within execution precision. A nonzero effect
supports the causal importance of the configured path under these controls; it
does not establish that the path is uniquely responsible for the behavior.

## Paper and author-code rule

The specification follows [Wang et al., Appendix B, Figure 11 and Algorithm 1](https://arxiv.org/pdf/2211.00593).
The author's [`path_patching` implementation](https://github.com/redwoodresearch/Easy-Transformer/blob/ea15315dd24481e9e2ac5c3ef335d82907a1dc34/easy_transformer/ioi_utils.py)
is useful for the two-pass receiver-capture/replay pattern, but freezes Q/K/V
activations of all heads and optionally freezes MLP outputs. Its receiver cache
is registered before its Q/K/V freeze hooks. This library adopts the requested
**head-output freezing with live MLP/LayerNorm computation**, and does not expose
those alternate author-code controls as silent defaults. Accordingly, this is an
implementation of that explicit IOI path-control rule rather than a claim of
bitwise reproduction of every author-code setting.

[Paper / source reference](../REFERENCES.md#causal-methods) ·
[Implementation](../../src/vlm_probing/causal/path_patching.py) ·
[Validation tests](../../tests/test_path_patching.py)

For a data-to-heatmap workflow with first-answer-token margins and per-pair
normalization, see the [path-patching notebook](../../demos/path_patching_demo.ipynb)
and the [data source index](../REFERENCES.md#datasets-and-evaluation-inputs).
The notebook records whether a paired input comes from an official image dataset
or an author-published VRUBench text example; it does not claim a full-dataset run.
