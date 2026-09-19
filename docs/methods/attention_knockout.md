# Attention Knockout

[Home](../../README.md) / [Methods](README.md) / Attention Knockout

Block selected query-to-key attention paths before softmax.

## Implementation

At an editable masked-logit site, set selected entries to negative infinity. The model's own softmax then excludes these edges and renormalizes remaining paths. Evaluate the resulting output score.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.causal.knockout(layers=[0], queries="text", keys="visual").run(
    inputs, metric=metric,
)
print(result.tensors["effect"])
```

## API

```text
probe.causal.knockout(*, layers=None, queries="text", keys="visual")
method.run(inputs, *, metric)
```

| Parameter | Meaning |
| --- | --- |
| `queries / keys` | Common token selectors defining blocked paths. |
| `layers` | Layers with explicitly editable masked-logit sites. |
| `metric` | Score for comparing edited and unedited runs. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`, per independently edited layer.

## Support and scope

The CPU example provides a real masked-logit site. Native HF adapters do not expose it. The kernel rejects a valid query whose every permitted key would be blocked.

[Paper / source reference](../REFERENCES.md#attention-knockout) ·
[Tensor implementation](../../src/vlm_probing/causal/attention_knockout.py) ·
[Shared result conventions](README.md#shared-conventions)
