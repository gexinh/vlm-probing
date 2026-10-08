# Logit Lens

[Home](../../README.md) / [Methods](README.md) / Logit Lens

Decode intermediate residuals with the model's final normalization and vocabulary head.

## Implementation

For a selected layer and token, compute `lm_head(final_norm(h))`. No parameters are trained. The final-layer result should match the model's own logits, up to floating-point batching effects.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.lens.logit(layers=[0, 1], tokens="last_prompt").run(inputs)
print(result.tensors["logits"].shape)
print(result.tensors["positions"])
```

## API

```text
probe.lens.logit(*, layers=None, tokens="last_prompt")
method.run(inputs)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | All residual layers by default; an integer or unique list selects a subset. |
| `tokens` | `"last_prompt"`, `"all"`, `"visual"`, `"text"`, positions, or a boolean mask. |

Returns a `ProbeResult`: `logits[L,N,V]` and `positions[N,2]`. `N` packs selected batch/token positions.

## Support and scope

Available on supported language-model residual sites and all eight VLM families,
including every Qwen3.5 residual layer. Readout is observational; a decoded token is not evidence of causal influence.

[Paper / source reference](../REFERENCES.md#lens-methods) ·
[Tensor implementation](../../src/vlm_probing/lenses/logit.py) ·
[Shared result conventions](README.md#shared-conventions)
