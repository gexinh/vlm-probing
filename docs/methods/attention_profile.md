# Attention Profile

[Home](../../README.md) / [Methods](README.md) / Attention Profile

Measure attention entropy, concentration, and mass assigned to token groups.

## Implementation

Capture the actual language self-attention probability matrices. Compute entropy in nats, squared-probability concentration, maximum probability, and summed mass for disjoint key groups. Statistics at unselected query positions are zero.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
layer = probe.describe()["methods"]["attention.profile"]["sites"][0]
result = probe.attention.profile(
    layers=[layer], queries="last_prompt", groups={"visual": "visual", "text": "text"},
).run(inputs)
print(result.tensors["group_mass"].shape)
print(result.tensors["valid_queries"])
```

## API

```text
probe.attention.profile(*, layers=None, queries="all", groups=None, include_attention=False)
method.run(inputs)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | Observable attention layers; Qwen3.5 standard-attention layers only. |
| `queries` | Common token selector for query positions. |
| `groups` | `{name: selector}` for disjoint key groups; defaults to visual/text when a visual layout exists. |
| `include_attention` | Also return raw `attention[L,B,H,Q,K]` for spatial maps or token-to-token comparisons. |

Returns a `ProbeResult`: `entropy`, `concentration`, and `max_probability`: `[L,B,H,Q]`; `group_mass[L,B,H,Q,G]`; `valid_queries[L,B,Q]`.

## Support and scope

Requires eager attention for the native HF adapters. Empty visual groups have zero mass. Attention statistics alone do not measure causal importance.

[Paper / source reference](../REFERENCES.md#supporting-primitives) ·
[Tensor implementation](../../src/vlm_probing/attention/profile.py) ·
[Shared result conventions](README.md#shared-conventions)
