# Attention Lens

[Home](../../README.md) / [Methods](README.md) / Attention Lens

Train separate vocabulary decoders for individual attention heads.

## Implementation

Capture each head immediately before `o_proj`, then project its contribution into residual coordinates. Train per-head affine vocabulary decoders jointly so the sum of their logits matches the model output distribution by KL divergence. Decoders start randomly by default; passing the native output-head weights reproduces the released author's initialization. Fixed unembedding attribution is a separate method.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.
The calibration example is sized for TinyModel; read the cost and scope notes before fitting a pretrained model.

```python
layer = probe.describe()["methods"]["lens.attention"]["sites"][0]
lens = probe.lens.attention(layers=[layer])
lens.fit(calibration, steps=20, lr=0.01)
result = lens.run(evaluation)
print(result.tensors["head_logits"].shape)
print(lens.losses[layer][-1])
```

## API

```text
probe.lens.attention(*, layers=None, tokens="last_prompt", binding=None)
method.fit(inputs, *, steps=100, lr=1e-3)
method.fit_batches(train_inputs, validation_inputs, directory, **training_options)
method.run(inputs)
method.save(directory)
method.load(directory)
```

| Parameter | Meaning |
| --- | --- |
| `layers` | Head sites; for Qwen3.5, standard-attention layers only. |
| `tokens` | Select matching teacher and head-output positions. |
| `steps / lr` | Calibration steps and learning rate. |
| `binding` | Artifact identities, as for Tuned Lens. |

Returns a `ProbeResult`: `head_logits[L,N,H,V]`, summed `logits[L,N,V]`, and `positions[N,2]`.

## Support and scope

Use the small CPU example to try calibration. Dense decoder storage is approximately `H * D_residual * V` parameters per layer, before gradients and optimizer state; full-vocabulary calibration on Qwen3.5-4B is expensive. No compressed-decoder substitute is used. Qwen3.5 captures gated head values.

The [text notebook](../../demos/lens_comparison_demo.ipynb) displays measured
readouts from trained GPT-2 decoders at layers 7 and 10, including every head.
Its default replay needs no weights; an opt-in rerun requires matching artifacts.
See the [training guide](../TRAINING.md) for persistent minibatch optimization,
validation and resuming. The demo records a 4,096-context calibration fit,
512 independent validation contexts, and the matching artifact identities.

[Paper / source reference](../REFERENCES.md#lens-methods) ·
[Tensor implementation](../../src/vlm_probing/lenses/attention.py) ·
[Shared result conventions](README.md#shared-conventions)
