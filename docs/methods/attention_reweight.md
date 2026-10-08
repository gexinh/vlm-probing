# Attention Reweighting

[Home](../../README.md) / [Methods](README.md) / Attention Reweighting

Rescale attention on selected key tokens and measure the output change.

This is a generic probability intervention, with no unique originating paper.
It is not a reproduction of PAI, VAR, or another complete decoding algorithm.

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

Use an adapter declaring probability sites actually consumed by `A @ V`.
Audited native eager Llama, Mistral, Qwen2, Qwen2-VL, Qwen2.5-VL, Qwen3-VL
and Qwen3.5 full-attention blocks expose those sites on Transformers 5.3;
the native ViT classifier also supports them. Diagnostic-only attention
outputs cannot implement the edit; the factory rejects them.

[Scope / related work](../REFERENCES.md#supporting-primitives) ·
[Tensor implementation](../../src/vlm_probing/attention/reweight.py) ·
[Shared result conventions](README.md#shared-conventions)
