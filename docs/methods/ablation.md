# Ablation

[Home](../../README.md) / [Methods](README.md) / Ablation

Replace selected residual activations with zeros, a reference mean, or a reference sample.

## Implementation

Build replacement vectors at a selected layer, write them into the model's actual continuation, and compare its score with the unedited run.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.causal.ablate(layers=[0], mode="zero").run(inputs, metric=metric)
resampled = probe.causal.ablate(layers=[0], mode="resample").run(
    inputs, reference=corrupt, metric=metric,
)
print(result.tensors["effect"], resampled.tensors["effect"])
```

## API

```text
probe.causal.ablate(*, layers=None, tokens="visual", mode="zero", mean_dims=(0,))
method.run(inputs, *, metric, reference=None, alignment="strict")
```

| Parameter | Meaning |
| --- | --- |
| `mode` | `"zero"`, `"mean"`, or `"resample"`. |
| `reference` | Required for mean/resample; native inputs that provide replacement activations. |
| `mean_dims` | Dimensions reduced to construct the mean, with singleton dimensions retained. |
| `alignment` | Resampled references must pass clean/corrupt alignment checks. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`, per independently edited layer.

## Support and scope

All residual layers in all seven VLM adapters. Resample uses the supplied reference directly; it does not draw random samples. Mean references must be unpadded. Ablation replaces vectors without deleting tokens or altering sequence length.

[Paper / source reference](../REFERENCES.md#generic-primitives) ·
[Tensor implementation](../../src/vlm_probing/causal/ablation.py) ·
[Shared result conventions](README.md#shared-conventions)
