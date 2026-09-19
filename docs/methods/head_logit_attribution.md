# Head Logit Attribution

[Home](../../README.md) / [Methods](README.md) / Head Logit Attribution

Project head contributions to vocabulary logits with a fixed final normalization scale.

## Implementation

Split the pre-`o_proj` value vector into query heads and apply the corresponding output-projection slices. Use the full final residual's normalization denominator and norm gain with the unembedding matrix. Biases and downstream responses to removing a head are excluded.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
layer = probe.describe()["methods"]["attention.head_logits"]["sites"][0]
result = probe.attention.head_logits(layers=[layer]).run(inputs)
print(result.tensors["head_logits"].shape)
```

## API

```text
probe.attention.head_logits(*, layers=None)
method.run(inputs)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | Layers with projected heads and an adapter-declared linear readout. |

Returns a `ProbeResult`: `head_logits[L,B,H,Q,V]` and `summed_logits[L,B,Q,V]`.

## Support and scope

All standard-attention layers in supported models, including gated Qwen3.5 heads. No fitting is needed. Intermediate-layer scores are direct projections, not a complete decomposition of final logits. Output size scales with heads, sequence length, and vocabulary; begin with one layer and a short prompt.

[Paper / source reference](../REFERENCES.md#head-attribution) ·
[Tensor implementation](../../src/vlm_probing/attention/head_logit_attribution.py) ·
[Shared result conventions](README.md#shared-conventions)
