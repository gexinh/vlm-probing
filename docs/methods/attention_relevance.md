# Attention Relevance

[Home](../../README.md) / [Methods](README.md) / Attention Relevance

Propagate positive gradient-weighted self-attention for a selected output score.

## Implementation

For each layer, compute `C = mean_heads(relu(A * dScore/dA))`, then update `R = R + C @ R`, starting from identity. This recurrence is not row-normalized. Gradients come from the same forward that produced the captured probabilities.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.attention.relevance(layers=[0, 1]).run(inputs, metric=metric)
print(result.tensors["relevance"].shape)
```

## API

```text
probe.attention.relevance(*, layers=None)
method.run(inputs, *, metric)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | Consecutive attention layers starting at zero. |
| `metric` | Differentiable TokenMargin, SequenceLogProb, or callable score. |

Returns a `ProbeResult`: `relevance[B,T,T]`, `layer_relevance[L,B,T,T]`, and `weighted_attention[L,B,T,T]`.

## Support and scope

Works with frozen model parameters by starting an activation graph at the declared embedding site. Requires eager attention; Qwen3.5-4B is unsupported. Implements the self-attention recurrence, not the full multimodal/cross-attention Chefer algorithm or LRP.

[Paper / source reference](../REFERENCES.md#attention-methods) ·
[Tensor implementation](../../src/vlm_probing/attention/relevance.py) ·
[Shared result conventions](README.md#shared-conventions)
