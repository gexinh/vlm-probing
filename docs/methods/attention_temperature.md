# Attention Temperature

[Home](../../README.md) / [Methods](README.md) / Attention Temperature

Change the sharpness of attention at selected query rows.

## Implementation

Divide the actual masked pre-softmax attention logits by a positive temperature, preserving causal/key masks, then replay the model and compare scores.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.attention.temperature(layers=[0], temperature=0.7).run(inputs, metric=metric)
print(result.tensors["effect"])
```

## API

```text
probe.attention.temperature(*, layers=None, queries="all", temperature=1.0)
method.run(inputs, *, metric)
```

| Parameter | Meaning |
| --- | --- |
| `queries` | Query positions to edit. |
| `temperature` | Finite positive scalar; values below 1 sharpen attention. |
| `metric` | Output score used for the intervention effect. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`, stacked over independently edited layers.

## Support and scope

Requires a custom masked-logit intervention site; the CPU example provides one. Native HF adapters do not expose this site. This is attention temperature, not token-sampling temperature, and is a generic intervention primitive.

[Paper / source reference](../REFERENCES.md#generic-primitives) ·
[Tensor implementation](../../src/vlm_probing/attention/temperature.py) ·
[Shared result conventions](README.md#shared-conventions)
