# Attention Reweighting

[Home](../../README.md) / [Methods](README.md) / Attention Reweighting

Rescale attention on selected key tokens and measure the output change.

## Implementation

At an editable probability site before `A @ V`, multiply selected query/key entries by a nonnegative weight. Optionally renormalize the entire row, then replay the model.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.attention.reweight(layers=[0], weight=2.0).run(inputs, metric=metric)
print(result.tensors["effect"])
```

## API

```text
probe.attention.reweight(*, layers=None, queries="text", keys="visual",
                        weight=2.0, renormalize=True)
method.run(inputs, *, metric)
```

| Parameter | Meaning |
| --- | --- |
| `queries / keys` | Common token selectors specifying the affected entries. |
| `weight` | Finite nonnegative multiplier. |
| `renormalize` | Preserve row probability sums; disabling it also changes output scale. |
| `metric` | Score to compare with the unedited model. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`, stacked over independently edited layers.

## Support and scope

Use the CPU example's explicit adapter. Native HF returned attentions are observational and cannot implement this edit; the factory raises CapabilityError there. A custom adapter must declare probability sites actually consumed by `A @ V`.

[Paper / source reference](../REFERENCES.md#head-attribution) ·
[Tensor implementation](../../src/vlm_probing/attention/reweight.py) ·
[Shared result conventions](README.md#shared-conventions)
