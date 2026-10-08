# Visualization

[Home](../README.md) / [Documentation](README.md)

The plotting functions accept `ProbeResult` objects or measured arrays and
return Matplotlib axes (or a figure for the path-diagram helpers). They operate
on results from any compatible model and
dataset; no model forward is hidden inside a plot.

```bash
pip install -e ".[visualization]"
```

```python
from vlm_probing import visualization as viz
```

## Lens readouts

```python
result = probe.lens.logit(layers=[0, 7, 15], tokens="text").run(inputs)
ax = viz.plot_lens_heatmap(
    result, target_token_id=answer_token_id,
    token_labels=selected_position_labels,
)
ax.figure.savefig("lens.png", bbox_inches="tight")
```

Rows retain the result's layer coordinates; columns are its packed selected
positions. With logits, choose a vocabulary `target_token_id`; the default
color is that token's one-based rank, using a logarithmic scale. Set
`value_kind="logit"` or `"probability"` for alternative readouts. A measured
two-dimensional rank array can be passed directly, with explicit
`layer_labels` and `token_labels`.

For Attention Lens, pass `tensor="head_logits", head=...` to select one head.
The plot selects `batch_index=0` by default. `annotations` may contain native
decoded fragments; plotting never translates tokens or filters the vocabulary.
Use shared `vmin`/`vmax` for comparable panels, and retain the same selected
target and positions. A fragment's token rank is not a complete word's
probability.

## Attention overlays

```python
result = probe.attention.grad_cam(
    layers=[last_layer], queries="last_prompt", keys="visual",
).run(inputs, metric=metric)

ax = viz.plot_attention_overlay(
    processed_image, result, tensor="cam",
    grid_shape=(grid_height, grid_width),
    visual_positions=visual_token_indices,
    query_positions=[prediction_position],
    bounding_boxes=[shirt_box_xywh],
)
ax.figure.savefig("attention.png", bbox_inches="tight")
```

`processed_image` is the image in the actual model input geometry, after its
resize, padding, or crop. Grid shape and patch indices must come from the
processor/layout; the library does not guess a square grid from token count or
align a processed map to the original photograph. Bounding boxes use pixel
`(x, y, width, height)` coordinates in that same displayed image.

The overlay accepts a two-dimensional spatial map, a vector with an explicit
`grid_shape`, or a result tensor with selected query/key coordinates. Raw
attention heads are averaged for display. Use `signed=True` for signed
attributions such as ATTATTR; a measured zero map stays zero.

```python
fig, axes = viz.plot_attention_comparison(
    processed_image,
    {"Grad-CAM": cam_result, "ATTATTR": attr_result, "TAM": tam_result},
    grid_shape=(grid_height, grid_width),
    visual_positions=visual_token_indices,
    query_positions=[prediction_position],
    signed={"Grad-CAM": False, "ATTATTR": True, "TAM": False},
    columns=3, shared_scale=False,
)
fig.savefig("attention-comparison.png", bbox_inches="tight")
```

Independent panel scales compare spatial shape. `shared_scale=True` compares
raw magnitudes only when the underlying units are comparable. Titles identify
the supplied methods; captions should specify the model, selected answer score,
query, and image preprocessing.

## Causal grids and curves

```python
paths = probe.causal.path(senders=senders, receivers="residual").sweep(
    base_inputs, donor=reference_inputs, metric=metric,
)
grid, layers, heads = viz.coordinate_grid(
    paths.tensors["senders"], paths.tensors["effect"][:, 0],
)
ax = viz.plot_causal_heatmap(
    grid, layer_labels=layers, column_labels=heads,
    highlight_count=3, value_label="Answer-score change",
)
```

`coordinate_grid` places measured values at their actual coordinates.
Unmeasured cells remain NaN; no interpolation creates missing measurements.
The signed heatmap uses a zero-centered scale; `highlight_count` outlines
the strongest measured cells. A path sweep tests configured paths, not an
automatically discovered circuit.

```python
ax = viz.plot_intervention_curves(
    {"Image to question": image_question_effects,
     "Question to prediction": question_prediction_effects},
    x=window_centers, xlabel="Window center layer", ylabel="Answer-score change",
)
```

Pass arrays of measured effects. If instead plotting raw intervention scores,
supply the unmodified `baseline` and use `relative=True` for relative change.
Do not compare first-subword probability, complete-answer log probability, and
logit-margin effects as though they shared units.

## Inputs and connections

`plot_input(image, prompt=..., answer=..., bounding_boxes=...)` shows the
input and its semantic target. It accepts the same explicit image geometry
and box coordinates as overlays.

`plot_connections(edges, node_layers=..., top_k=...)` draws supplied weighted
connections in layer order. Signed color and edge width encode scores; the
function neither computes edge attributions nor infers functional head names.

For measured sender-to-final-residual sweeps,
`plot_head_paths(senders, effects, pairs=..., model_name=..., top_k=5)` returns
a figure of the selected scored routes.
`plot_receiver_paths(artifact, model_name=...)` returns a figure of measured
receiver Q/K/V route scans. These diagrams display supplied interventions;
they do not recover untested edges or infer functional head labels.

See the [demo notebooks](../demos/README.md) for end-to-end plots from actual
measured arrays, sources, and inputs. Figures can be saved as PNG, SVG, or PDF
with Matplotlib's normal `savefig` interface.
