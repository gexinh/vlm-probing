# Attention Knockout

[Home](../../README.md) / [Methods](README.md) / Attention Knockout

Block selected attention paths before softmax and measure the change in model
output. Information travels **from source key positions to receiver query
positions**: `keys="visual", queries=question_mask` tests image → question.

## Implementation

At an editable masked-logit site, set selected query/key entries to negative
infinity. The model's own softmax then excludes these edges and renormalizes
remaining paths. This is an actual forward intervention; attention weights are
recomputed before value aggregation.

With `joint=True`, all selected layers are blocked in **one forward**. Later
layers compute new Q/K/V from the already changed residual stream. This is the
setting used for the paper's nine-layer windows. The default `joint=False`
measures each selected layer independently.

## Example

Use the [small CPU setup](README.md#example-setup) first. For a VLM, use its
[model guide](../models/README.md) and choose layers reported by `describe()`.

```python
result = probe.causal.knockout(layers=[0], queries="text", keys="visual").run(
    inputs, metric=metric,
)
print(result.tensors["effect"])
```

For a nine-layer window centered on decoder layer 12:

```python
window = list(range(8, 17))
method = probe.causal.knockout(
    layers=window, queries=question_mask, keys="visual", joint=True,
)
result = method.run(inputs, metric=fixed_answer_probability)
relative_change = 100 * result.tensors["effect"] / result.tensors["baseline_score"]
```

`question_mask` is a boolean `[batch, expanded_sequence]` mask for the original
question only. `"text"` also includes system text, answer instructions, and chat
markers, so it is not interchangeable with the question mask. Images are keys
and question positions are queries in this example.

See the executed [GQA notebook](../../demos/attention_knockout_demo.ipynb) for
two original images with target bounding boxes, exact prompts, model baselines,
three information-flow scans, and target-region versus other-region scans.

## API

```text
probe.causal.knockout(*, layers=None, queries="text", keys="visual", joint=False)
method.run(inputs, *, metric)
method.forward(inputs, *, capture=())  # joint=True only
```

| Parameter | Meaning |
| --- | --- |
| `queries / keys` | Common token selectors defining blocked paths. |
| `layers` | Layers with explicitly editable masked-logit sites. |
| `joint` | Block all configured layers in a single forward; default scans layers independently. |
| `metric` | Score for comparing edited and unedited runs. |
| `capture` | Hook sites to retain during an optional joint intervened `forward`. |

`run` returns a `ProbeResult`: `baseline_score`, `intervention_score`, and `effect`.
For `joint=True`, the score shape is the metric's original shape (normally
`[batch]`); for the default, results are stacked per independently edited layer.
`forward` returns the actual intervened `ForwardTrace` without repeating the
baseline, which is useful for a caller who already has an unmodified score.

## Support and scope

Audited Transformers 5.3 eager Llama (including **LLaVA-1.5**), Mistral,
Qwen2, Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3.5 full-attention blocks
expose editable masked scores. Use `probe.describe()` to check the supplied
model. Other backends require an explicit adapter with `ModelSpec.attention_scores`.
Fused SDPA/Flash attention does not expose this intervention site.

The kernel rejects a valid query whose every permitted key would be blocked.
Selectors use the model's expanded sequence, including its image tokens; the
demo verifies all 576 LLaVA visual positions and the original question span.
For multi-layer interventions, use `joint=True` so downstream attention scores
are recomputed instead of replaying their baseline values.

The demo fixes the first **generated subword** of the correct baseline answer
throughout the scan: `Yellow` starts with `Y`, whereas `Black` is one token. Its
probability is taken over the full vocabulary. Each point is
`100 * (p_intervention - p_baseline) / p_baseline`, matching the paper's Sections
5–6 and Figures 3–5. These are two example-level experiments, not estimates of
the original dataset-wide trends.

[Paper / source reference](../REFERENCES.md#causal-methods) ·
[Tensor implementation](../../src/vlm_probing/causal/attention_knockout.py) ·
[Shared result conventions](README.md#shared-conventions)
