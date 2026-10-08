# Residual Steering

[Home](../../README.md) / [Methods](README.md) / Residual Steering

Add a supplied direction to selected residuals and measure the output change.

## Implementation

Apply `h + strength * direction` at chosen token positions, optionally rescaling edited vectors to preserve their original norm, then continue the model.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
direction = torch.ones(model.embedding.embedding_dim)  # TinyModel demonstration direction
result = probe.causal.steer(layers=[0], strength=0.1).run(
    inputs, direction=direction, metric=metric,
)
print(result.tensors["effect"])
```

## API

```text
probe.causal.steer(*, layers=None, tokens="visual", strength=1.0, preserve_norm=False, joint=False)
method.run(inputs, *, direction, metric)
method.forward(inputs, *, direction, capture=())       # joint=True
method.generate(inputs, *, direction, max_new_tokens=64, do_sample=False)
probe.causal.vsv(positive, negative, *, layers=None)
```

| Parameter | Meaning |
| --- | --- |
| `direction` | A broadcastable tensor in residual coordinates, or `{layer: tensor}`. |
| `strength` | Finite signed scale for the supplied direction. |
| `preserve_norm` | Rescale the edited activation vectors to their original norms. |
| `joint` | Edit every chosen layer in the same live forward. |
| `metric` | Output score used for the intervention effect. |

Returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`.
The default measures layers independently. Use `joint=True` to modify all
selected residual sites in one live forward; each later layer sees the earlier
intervention. The joint configuration also supports `.forward(..., capture=...)`
and greedy `.generate(...)`.

## Per-image visual steering (VSV)

The [executed Steering / VSV notebook](../../demos/steering_vsv_demo.ipynb)
contains two COCO 2014 image-description examples with LLaVA-1.5-7B. It shows
full input prompts, baseline and steered captions, token-ranking heatmaps at
identical baseline prefixes, and a fixed-prefix strength-response curve. It
keeps a diagnostic `λ=0.17` alongside an illustrative smaller `λ=0.02`.
These scales are not calibrated optima for the raw-residual implementation. The raw-residual `0.17`
run produces repetitive captions on both images; the smaller scale produces
coherent captions and changes some grounded details. This does not establish
the full VISTA framework's dataset-level effectiveness.

Those historical runs predate the strict input-alignment checks in the new
three-model comparison. Its no-image input keeps the positive input's exact
nonvisual token IDs, including processor-induced boundary whitespace. Native
visual wrappers are removed as a complete span; direction construction then
uses the same text tokens in both conditions.

```python
from vlm_probing.causal.visual_steering import VisualSteering

# Prepare the same question with and without its image. No answer text is used.
vsv = VisualSteering.from_inputs(probe, image_inputs, text_only_inputs)
method = vsv.configure(strength=0.02, tokens="prediction")  # illustrative; sweep for your model
output = method.generate(
    image_inputs, direction=vsv.directions,
    max_new_tokens=144, do_sample=False,
)
caption = processor.tokenizer.decode(output.tensors["generated_ids"][0],
                                     skip_special_tokens=True)
```

For every layer, the direction is the difference between the **last context
token's block-output residual**, with and without visual input. The joint
intervention adds `λ * direction` to live residual vectors and restores their
original norms. Directions remain fixed across generation. The demo applies
the intervention to the original context-end prediction position and generated
positions in uncached full-sequence replays. Other prompt and visual positions
are unchanged. Teacher-forced comparisons supply a fixed original prompt mask
so appending caption tokens does not move the context boundary. An all-token
scope experiment is retained as an explicitly different diagnostic.

This follows VISTA paper **Equations 4–6**. The released author implementation
also adds a PCA component and edits **MLP outputs**, with normalized directions
and a cosine-dependent scale. Those operations differ from the paper's residual
formula and are not silently mixed into this implementation. The notebook is
**VSV only**; it does not implement the separate SLA logit augmentation, so it
is not labeled a complete VISTA reproduction.

Steering scales depend on the intervention rule and model. For the raw
direction, the relative additive perturbation is
`strength * ||direction|| / ||residual||`; identical strengths can therefore
produce different effects across layers and models. The author's normalized
MLP-output implementation uses a different scale. Its `λ=0.17` is not a
portable default for this raw block-residual demo. Compare a small strength
sweep, inspect both visual claims and repetition, and retain complete outputs.

The [technical report](https://github.com/gexinh/vlm-probing/releases/latest/download/vlm-probing-technical-report.pdf)
also compares the same COCO image in LLaVA, SmolVLM and Qwen3-VL. The portable
[Steering demo](../../demos/steering_vsv_demo.ipynb) includes saved measurements;
its manifest retains the model, input, and intervention identities.

## Support and scope

All declared residual layers in supported adapters. Generic steering consumes
a supplied direction; `VisualSteering.from_inputs` constructs per-image,
training-free VSV directions. Neither operation trains a steering classifier or
changes model weights.

[Paper / source reference](../REFERENCES.md#causal-methods) ·
[Tensor implementation](../../src/vlm_probing/causal/steering.py) ·
[Shared result conventions](README.md#shared-conventions)
