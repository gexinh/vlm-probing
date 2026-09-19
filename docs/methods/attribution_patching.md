# Attribution Patching

[Home](../../README.md) / [Methods](README.md) / Attribution Patching

Approximate a patch effect using activation differences and a receiver-side gradient.

## Implementation

At each selected residual site, differentiate the batch-summed score with respect to the receiver activation, then multiply elementwise by `source - receiver`. Sum over hidden dimensions for token scores and over selected tokens for a layer estimate.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
approximate = probe.causal.attribute(layers=[0]).run(corrupt, source=inputs, metric=metric)
exact = probe.causal.patch(layers=[0]).run(corrupt, source=inputs, metric=metric)
print(approximate.tensors["estimated_effect"], exact.tensors["effect"].sum(-1))
```

## API

```text
probe.causal.attribute(*, layers=None, tokens="visual")
method.run(inputs, *, source, metric, alignment="strict")
```

| Parameter | Meaning |
| --- | --- |
| `source` | Aligned donor inputs defining the activation difference. |
| `metric` | Differentiable output score. |
| `layers / tokens / alignment` | Same selection and alignment semantics as activation patching. |

Returns a `ProbeResult`: `token_scores[L,B,T]` and batch-summed `estimated_effect[L]`.

## Support and scope

All residual layers in all seven VLM adapters. This first-order estimate can differ from the exact intervention, especially for large changes. Use exact patching to validate important sites.

[Paper / source reference](../REFERENCES.md#attribution-patching) ·
[Tensor implementation](../../src/vlm_probing/causal/attribution_patching.py) ·
[Shared result conventions](README.md#shared-conventions)
