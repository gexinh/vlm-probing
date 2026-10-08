# EAP-IG

[Home](../../README.md) / [Methods](README.md) / EAP-IG

Score transformer residual-message edges with **gradients integrated along the
input-embedding path**, then test selected edges with real forward interventions.
The [executed IOI notebook](../../demos/eap_ig_demo.ipynb) uses native GPT-2
small and two examples from the paper authors' published dataset.

## Original input-path implementation

For clean input embeddings `x`, corrupted embeddings `x'`, and a source node
`u`, keep the endpoint difference `delta_u = z'_u - z_u` fixed. Run the model
on `x' + alpha * (x - x')`, obtain the loss gradient at each destination input
`v`, and compute:

```text
EAP(u → v)    = delta_u · grad_v loss(x)
EAP-IG(u → v) = delta_u · mean_alpha grad_v loss(x' + alpha * (x - x'))
loss         = -mean(task metric over examples)
```

Scores sum over all aligned sequence positions and residual dimensions. A
positive score estimates an increase in loss when that edge is corrupted; it is
not a probability or proof that the head has a particular human-readable role.
Unlike input-feature IG, summing these graph-edge scores has no completeness
guarantee.

```python
from vlm_probing import Prober, TokenMargin

probe = Prober(model, processor=tokenizer)
method = probe.causal.eap_ig(graph="transformer", steps=5)
scores = method.run(
    clean_inputs,
    donor=corrupted_inputs,
    metric=TokenMargin(positive=io_token_id, negative=subject_token_id),
    alignment="position",
)
print(scores.tensors["edge_scores"])  # original EAP-IG
print(scores.tensors["eap_scores"])   # clean-endpoint EAP comparison
```

The current audited native graph supports standard `GPT2LMHeadModel`
architectures. GPT-2 small has **32,491 independently replaceable directed
edges**: input embeddings, individual projected head outputs, and MLP outputs
feed each downstream head's separate pre-LayerNorm Q/K/V input, each MLP input,
and the final residual input. The provider splits packed QKV projections without
changing weights. MLP receiver inputs are separate branch copies so their
gradients exclude the residual skip branch.

`quadrature="left"` uses `alpha = 0, 1/m, …, (m-1)/m`, matching the
[author implementation at the pinned revision](https://github.com/hannamw/EAP-IG/blob/e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd/src/eap/attribute.py#L149).
`quadrature="right"` uses `1/m, …, 1`, matching [paper equation 3](https://arxiv.org/html/2403.17806v2#S3).
Both interpolate **inputs**, not internal edge activations.

## Actual circuit evaluation

```python
evaluation = method.evaluate(
    clean_inputs,
    donor=corrupted_inputs,
    metric=metric,
    attribution=scores,
    budgets=[0, 100, 500, 1000, len(method.graph.edge_names)],
    rankings=("eap", "eap_ig"),
    strategy="greedy",
)
print(evaluation.tensors["faithfulness"])
```

At each destination, excluded edges receive cached corrupted source outputs;
retained edges receive outputs from the **current intervened forward**. Every
excluded real edge in the full graph is corrupted. Backward greedy search uses
absolute scores and starts at the readout; pruning removes nodes without an
input-to-readout path. `strategy="topk"` is an explicit alternative.

`faithfulness = (circuit margin - corrupted margin) / (clean margin - corrupted
margin)`. Zero means the corrupted reference; one means the clean margin.
Negative values and values above one are allowed. Increasing the edge budget
does not guarantee a monotonic curve. Ineligible near-zero denominators are
marked by `eligible=False`; their displayed zero is not evidence of failure.

Results include raw margins, requested and actual retained edge counts, retained
masks, and all-retained/none-retained endpoint errors. To check approximation
quality on selected edges, use
`method.patch_edges(clean_inputs, donor=..., metric=..., excluded_edges=[...])`.
Each selected edge is corrupted independently in a real forward pass.

## Explicit activation-space variant

The previous caller-defined kernel remains available:

```python
method = probe.causal.eap_ig(graph="activation", edges=edge_names, steps=8)
result = method.run(receiver_inputs, source=source_inputs, metric=metric)
```

This route requires an adapter declaring independently replaceable edge sites
in `ModelSpec.edges`. It interpolates selected **internal edge activations
jointly**, with trapezoidal quadrature. It is a separate activation-space variant,
not the original input-path EAP-IG algorithm. The default `graph="activation"`
preserves the earlier public interface; choose `graph="transformer"` explicitly
for the native original method.

## Scope

The native original method currently supports text-only GPT-2 graphs; VLM graph
support is not claimed. New model families need audited source decomposition,
independent destination inputs, and no-op/full-knockout fidelity checks. Scores
identify candidate connections. Functional head names need further experiments.

[Paper](https://arxiv.org/abs/2403.17806) ·
[Author dataset](https://github.com/hannamw/eap-ig-faithfulness/blob/d42fea6aa2c7acca39007c9c94923770669cb9ae/data/ioi/gpt2.csv) ·
[Native graph implementation](../../src/vlm_probing/causal/transformer_eap_ig.py) ·
[Demo provenance](../../demos/README.md)
