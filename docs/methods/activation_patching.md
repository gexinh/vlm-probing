# Activation Patching

[Home](../../README.md) / [Methods](README.md) / Activation Patching

Replace selected receiver residuals with source residuals and measure the causal effect.

## Implementation

Run clean/source and corrupt/receiver inputs, validate their token alignment, copy selected activations at one layer, and re-run the receiver. A layer sweep performs independent experiments rather than patching every selected layer together.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.causal.patch(layers=[0, 1], tokens="visual").run(
    corrupt, source=inputs, metric=metric,
)
print(result.tensors["effect"])
```

## API

```text
probe.causal.patch(*, layers=None, tokens="visual")
method.run(inputs, *, source, metric, alignment="strict")
```

| Parameter | Meaning |
| --- | --- |
| `layers / tokens` | Residual sites and positions to replace. |
| `inputs / source` | Receiver and donor model inputs. |
| `metric` | One-token margin, teacher-forced sequence score, or callable. |
| `alignment` | `"strict"` checks token IDs, masks, and declared grid/position metadata; `"position"` relaxes only token-ID equality. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect = intervention - baseline`: `[L,B]` for a per-example metric.

## Support and scope

Supported language-model residual layers and all eight VLM families. Source and receiver use the same expanded coordinates; arbitrary position remapping is not implemented. Alignment does not by itself establish semantic equivalence between image regions.

[Paper / source reference](../REFERENCES.md#causal-methods) ·
[Tensor implementation](../../src/vlm_probing/causal/activation_patching.py) ·
[Shared result conventions](README.md#shared-conventions)
