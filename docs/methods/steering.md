# Residual Steering

[Home](../../README.md) / [Methods](README.md) / Residual Steering

Add a supplied direction to selected residuals and measure the output change.

## Implementation

Apply `h + strength * direction` at chosen token positions, optionally rescaling edited vectors to preserve their original norm, then continue the model.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
direction = torch.ones(model.embedding.embedding_dim)  # TinyModel demonstration direction
result = probe.causal.steer(layers=[0], strength=0.1).run(
    inputs, direction=direction, metric=metric,
)
print(result.tensors["effect"])
```

## API

```text
probe.causal.steer(*, layers=None, tokens="visual", strength=1.0, preserve_norm=False)
method.run(inputs, *, direction, metric)
```

| Parameter | Meaning |
| --- | --- |
| `direction` | A broadcastable tensor in residual coordinates, or `{layer: tensor}`. |
| `strength` | Finite signed scale for the supplied direction. |
| `preserve_norm` | Rescale the edited activation vectors to their original norms. |
| `metric` | Output score used for the intervention effect. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`, per independently edited layer.

## Support and scope

All residual layers in all seven VLM adapters. The method consumes a direction; it does not learn a visual/semantic direction or reproduce a complete steering paper's training recipe.

[Paper / source reference](../REFERENCES.md#steering) ·
[Tensor implementation](../../src/vlm_probing/causal/steering.py) ·
[Shared result conventions](README.md#shared-conventions)
