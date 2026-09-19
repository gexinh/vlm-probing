# Attention Rollout

[Home](../../README.md) / [Methods](README.md) / Attention Rollout

Compose normalized self-attention transitions across consecutive layers.

## Implementation

Fuse heads, optionally add identity for the residual path, normalize each row, and multiply transitions in forward layer order. Padding is excluded.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.attention.rollout(layers=[0, 1]).run(inputs)
print(result.tensors["rollout"].shape)
```

## API

```text
probe.attention.rollout(*, layers=None, head_reduction="mean", residual=True)
method.run(inputs)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | Consecutive available attention layers starting at zero; defaults to the available prefix. |
| `head_reduction` | `"mean"`, `"max"`, or `"min"`; default `"mean"` follows the standard convention. |
| `residual` | Add the identity before row normalization. |

Returns a `ProbeResult`: Final `rollout[B,T,T]` and `layer_rollouts[L,B,T,T]`.

## Support and scope

Requires eager self-attention. Unavailable on Qwen3.5-4B because linear-attention gaps cannot be skipped. It summarizes attention transitions and omits values, MLPs, and visual injection edges.

[Paper / source reference](../REFERENCES.md#rollout) ·
[Tensor implementation](../../src/vlm_probing/attention/rollout.py) ·
[Shared result conventions](README.md#shared-conventions)
