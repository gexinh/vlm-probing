# Tuned Lens

[Home](../../README.md) / [Methods](README.md) / Tuned Lens

Fit one affine translator per residual layer before applying the model's frozen readout.

## Implementation

The translator starts at identity through `h + affine(h)`. Calibration minimizes `KL(model output || lens output)` at the selected positions, with no next-token target shift. Model parameters are not optimized.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.
The calibration example is sized for TinyModel; read the cost and scope notes before fitting a pretrained model.

```python
binding = {"model_id": "tiny-seed7", "tokenizer_id": "integer-demo",
           "readout_id": "tiny.norm+head", "calibration_id": "demo-batch-v1"}
lens = probe.lens.tuned(layers=[0], binding=binding)
lens.fit(calibration, steps=20, lr=0.01)
result = lens.run(evaluation)
print(lens.losses[0][0], lens.losses[0][-1])
lens.save("outputs/tuned")
restored = probe.lens.tuned(layers=[0], binding=binding).load("outputs/tuned")
```

## API

```text
probe.lens.tuned(*, layers=None, tokens="last_prompt", binding=None)
method.fit(inputs, *, steps=100, lr=1e-3)
method.fit_batches(train_inputs, validation_inputs, directory, **training_options)
method.run(inputs)
method.save(directory)
method.load(directory)
```

| Parameter | Meaning |
| --- | --- |
| `layers / tokens` | Residual layers and calibration/evaluation token selection. |
| `steps / lr` | Optimizer steps and learning rate for the translator. |
| `binding` | Model, tokenizer, readout, and calibration identities; required to save portable artifacts. |

Returns a `ProbeResult`: `logits[L,N,V]`, `positions[N,2]`, and calibration losses in `lens.losses[layer]`.

## Support and scope

Supported language-model residual sites and all eight VLM families. `fit` accepts one batch; repeated calls retain translators but restart the optimizer. Use held-out data to measure calibration quality. See [artifact identity](../API.md#fitted-lens-artifacts).

Use `fit_batches` for persistent Adam, validation and resume across a dataset;
see the [training guide](../TRAINING.md). The
[text notebook](../../demos/lens_comparison_demo.ipynb) displays readouts from
verified author GPT-2 weights. The [VLM notebook](../../demos/vlm_lens_demo.ipynb)
uses separately trained LLaVA translators and independent GQA examples.
Both replay saved measurements by default; loading compatible artifacts is
an explicit model-rerun option.

[Paper / source reference](../REFERENCES.md#lens-methods) ·
[Tensor implementation](../../src/vlm_probing/lenses/tuned.py) ·
[Shared result conventions](README.md#shared-conventions)
