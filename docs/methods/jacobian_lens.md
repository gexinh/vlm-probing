# Jacobian Lens

[Home](../../README.md) / [Methods](README.md) / Jacobian Lens

Estimate a layer-to-final-residual Jacobian, transport representations, and decode them.

## Implementation

Re-run the model with a differentiable residual intervention at the selected layer. Sum final-residual cotangents over valid target positions, average derivatives over selected source positions and prompts, then store `J[out,in]`. Evaluation applies `J @ h` followed by the model readout.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.
The calibration example is sized for TinyModel; read the cost and scope notes before fitting a pretrained model.

```python
lens = probe.lens.jacobian(layers=[0], tokens="all")
lens.fit(calibration, skip_first=0, exclude_last=False, max_sources=4)
result = lens.run(evaluation)
print(result.tensors["transported"].shape)
print(result.tensors["logits"].shape)
```

## API

```text
probe.lens.jacobian(*, layers=None, tokens="all", binding=None)
method.fit(inputs, *, skip_first=16, exclude_last=True,
           target_mask=None, max_sources=None, seed=0)
method.run(inputs)
method.save(directory)
method.load(directory)
```

| Parameter | Meaning |
| --- | --- |
| `tokens` | Source positions, intersected with valid positions and calibration exclusions. |
| `skip_first / exclude_last` | Exclude early/final valid positions; override for short prompts. |
| `target_mask` | Optional boolean `[B,T]` target-position mask. |
| `max_sources / seed` | Subsample source positions deterministically; does not reduce the number of output-dimension backward passes. |
| `binding` | Artifact identities, as for Tuned Lens. |

Returns a `ProbeResult`: `transported[L,N,D_final]`, `logits[L,N,V]`, and `positions[N,2]`.

## Support and scope

All seven VLM adapters. Exact fitting needs one backward pass per final hidden dimension, so the small CPU example is the starting point. Repeated `fit` replaces the estimate. Do not fit under `torch.inference_mode()`. The transport is linear and has no fitted intercept; it is not a full nonlinear causal explanation.

[Paper / source reference](../REFERENCES.md#jacobian-lens) ·
[Tensor implementation](../../src/vlm_probing/lenses/jacobian.py) ·
[Shared result conventions](README.md#shared-conventions)
