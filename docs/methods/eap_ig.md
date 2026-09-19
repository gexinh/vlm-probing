# EAP-IG (explicit edge activations)

[Home](../../README.md) / [Methods](README.md) / EAP-IG (explicit edge activations)

Integrate gradients along a simultaneous interpolation of declared edge messages.

## Implementation

Capture clean/corrupt activations on independently replaceable computational edges. Interpolate all selected edge messages jointly, re-run the graph, and accumulate gradients with trapezoidal quadrature. Multiply by each edge's activation difference to obtain its score.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.causal.eap_ig(steps=8).run(corrupt, source=inputs, metric=metric)
print(result.tensors["edge_scores"])
print(result.tensors["completeness_error"])
```

## API

```text
probe.causal.eap_ig(*, edges=None, steps=32)
method.run(inputs, *, source, metric, alignment="strict")
```

| Parameter | Meaning |
| --- | --- |
| `edges` | Names from ModelSpec.edges; defaults to all declared edges. |
| `steps` | Positive number of integration intervals; executes `steps + 1` gradient evaluations. |
| `source / metric / alignment` | Aligned donor inputs and a differentiable batch-summed score. |

Returns a `ProbeResult`: `edge_scores[E]`, endpoint scores, total effect, and quadrature `completeness_error`.

## Support and scope

Requires a custom adapter with equally shaped, independently editable edge messages; the CPU example provides them. This activation-space variant does not implement the upstream input-embedding interpolation algorithm, automatically extract circuits, or evaluate circuit faithfulness.

[Paper / source reference](../REFERENCES.md#eap-ig) ·
[Tensor implementation](../../src/vlm_probing/causal/eap_ig.py) ·
[Shared result conventions](README.md#shared-conventions)
