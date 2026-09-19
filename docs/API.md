# Public API: v0.2.0

`Prober(model, processor=None, adapter="auto", result_device="cpu")` binds the
three method collections. Factories configure a reusable method; `run` executes
it on native model kwargs or `ProbeInputs`. The examples below use the README's
`probe`, `clean`, `corrupt`, and `metric`. Layer indices are zero-based decoder
block outputs. Omitting `layers` selects all sites available for that method.

## Lenses

```python
probe.lens.logit(layers=[0, 1], tokens="last_prompt").run(clean)
probe.lens.embed(tokens="visual").run(clean, top_k=3)

probe.lens.tuned(layers=[0]).fit(clean, steps=20, lr=0.01).run(clean)
probe.lens.attention(layers=[0]).fit(clean, steps=20, lr=0.01).run(clean)
probe.lens.jacobian(layers=[0], tokens="all").fit(
    clean, skip_first=0, exclude_last=False,
).run(clean)

scope = probe.lens.patchscope(
    layers=[0], source_position=0, target_layer=0, target_position=2,
)
scope.run(clean, target_inputs=clean)
```

Logit/Tuned/Jacobian read residual block outputs. Embed reads the declared
embedding-space site, normally the first decoder block's input after multimodal
merging. Attention Lens reads each head's projected contribution in residual
coordinates; the current public decoder initialization is random. The tensor
class also accepts explicit unembedding initialization.

Tuned/Attention calibration uses the configured token mask and the full model
output distribution at the same positions; there is no next-token target shift.
For Jacobian, `tokens` selects source positions and padding is excluded. The
reference defaults exclude the first 16 valid tokens and the final valid token;
override these for short sequences. `target_mask` and `max_sources` are supported.
Do not pass a second `source_mask`/`valid_mask`: the bound method supplies them.

`fit` accepts one batch. Use separate calibration and evaluation examples in real
experiments. Tuned/Attention retain learned weights across repeated fits but reset
the optimizer; Jacobian replaces its previous estimate. `save(directory)` and
`load(directory)` use one file per layer, validating the explicit identity binding
shown in the README. `.fit` and `.load` return the bound object; fitted losses are
in `.losses[layer]` for Tuned/Attention.

Patchscope accepts `target=another_prober`. A different model instance requires
`mapping=callable`, even if dimensions match. Positions must be valid in every
example and batch sizes must match. Output is teacher-forced target logits;
free generation is outside this release.

## Causal probes

```python
probe.causal.patch(layers=[0], tokens="visual").run(
    corrupt, source=clean, metric=metric,
)
probe.causal.ablate(layers=[0], mode="zero").run(clean, metric=metric)
probe.causal.ablate(layers=[0], mode="mean", mean_dims=(0,)).run(
    clean, reference=corrupt, metric=metric,
)
probe.causal.ablate(layers=[0], mode="resample").run(
    clean, reference=corrupt, metric=metric,
)
probe.causal.attribute(layers=[0]).run(corrupt, source=clean, metric=metric)
probe.causal.steer(layers=[0], strength=0.1).run(
    clean, direction=torch.ones(6), metric=metric,
)
probe.causal.knockout(layers=[0], queries="text", keys="visual").run(clean, metric=metric)
probe.causal.eap_ig(steps=32).run(corrupt, source=clean, metric=metric)
```

`patch`, `ablate`, `attribute`, and `steer` default to `tokens="visual"`. Source and
reference activations are captured automatically. Layer interventions are
independent. `resample` uses the supplied reference batch; it does not sample
internally. Mean references must be unpadded; use tensor kernels for custom
masked reference distributions. Choose independent reference data and appropriate
controls for actual experiments.

`direction` may be a broadcastable tensor or `{layer: tensor}`. The method does
not estimate directions. `preserve_norm=True` follows the tensor kernel's complete
activation-vector norm convention.

`attribute` differentiates the batch-summed metric and returns token scores and
a summed estimated effect per layer. It is not an exact patch measurement.
`eap_ig` jointly interpolates declared edge messages with matching tensor shapes;
it returns per-edge scores, endpoint scores, and quadrature completeness error.
Its target is also batch-summed. These gradient methods assume independent batch
examples, as in a decoder operating in evaluation mode.

### Alignment

`alignment="strict"` checks valid/visual/prompt masks, expanded token IDs, and
adapter-declared grid/position metadata. The same positions are used in source
and receiver. Different sequence layouts are rejected.

For intentionally different text with verified positional correspondence,
`alignment="position"` relaxes only token-ID equality. Layout/grid checks still
apply. General position remapping is not implemented, and neither mode proves
semantic correspondence between image regions.

## Attention

```python
probe.attention.profile(layers=[0, 1], queries="last_prompt").run(clean)
probe.attention.rollout(layers=[0, 1], head_reduction="mean", residual=True).run(clean)
probe.attention.relevance(layers=[0, 1]).run(clean, metric=metric)
probe.attention.head_logits(layers=[1]).run(clean)
probe.attention.reweight(layers=[0], queries="text", keys="visual", weight=2.).run(
    clean, metric=metric,
)
probe.attention.temperature(layers=[0], queries="text", temperature=0.7).run(
    clean, metric=metric,
)
```

Profile defaults to disjoint visual/text groups when visual metadata is available.
Set `groups={name: token_selector}` for custom groups. Unselected query statistics
are zero; `valid_queries` identifies the selected rows.

Rollout and relevance need consecutive layers starting at zero and square
self-attention. Relevance differentiates actual attention tensors from the same
forward. Frozen model weights work when the adapter exposes an embedding site
where an activation graph can begin.

Head attribution uses the full final residual's fixed normalization scale and an
adapter-declared linear readout. Bias and downstream responses to head removal
are excluded. Intermediate-layer head scores are direct projections in this
readout convention, not a full decomposition of final logits.

Reweighting defaults to `renormalize=True`; disabling it changes output scale as
well as attention distribution. Temperature modifies attention softmax, not
next-token sampling. Both write into real computation sites and report independent
per-layer effects. All three target-dependent operations require `metric=`.

## Metrics and explicit layouts

```python
from vlm_probing import ProbeInputs, TokenLayout, TokenMargin, SequenceLogProb

# All masks: bool [batch, expanded_sequence].
# expanded_ids: long IDs aligned with output logits, including visual slots.
inputs = ProbeInputs(model_kwargs, TokenLayout(
    valid=valid, visual=visual, prompt=prompt, token_ids=expanded_ids,
))
single_token_score = TokenMargin(positive=42, negative=73, position="last_prompt")
answer_score = SequenceLogProb(answer_mask, reduction="sum")
```

`TokenMargin` needs exactly one prediction position per example. `SequenceLogProb`
shifts logits by one and scores only the selected answer tokens; EOS counts only
if selected. Visual tokens and padding cannot be answer targets. Each scored
answer needs a preceding prediction position. Score separate complete
prompt+candidate inputs for different candidate answers.

Custom metrics may be `callable(logits) -> scalar_or_batch_tensor`. Preserve the
computation graph for gradient methods; do not use `.item()` or detach. Objects
with `score(logits, layout)` can access resolved token metadata.

## Result coordinates

| Method | Main output |
| --- | --- |
| Logit, Tuned, Jacobian | `logits[L,N,V]`, `positions[N,2]`; Jacobian also `transported[L,N,D]` |
| Embed | `token_ids[1,N,K]`, `similarities[1,N,K]`, norms and positions |
| Attention Lens | `head_logits[L,N,H,V]`, summed `logits[L,N,V]`, positions |
| Patchscope | Target `logits[L,B,T_target,V]` |
| Exact interventions | `baseline_score`, `intervention_score`, `effect`: `[L,B]` or `[L]` |
| Attribution patching | `token_scores[L,B,T]`, `estimated_effect[L]` |
| EAP-IG | `edge_scores[E]`, scalar endpoint/effect/completeness values |
| Profile | `entropy[L,B,H,Q]`, `group_mass[L,B,H,Q,G]`, other statistics |
| Head logit attribution | `head_logits[L,B,H,Q,V]`, `summed_logits[L,B,Q,V]` |
| Rollout / relevance | Final `[B,T,T]` and intermediate `[L,B,T,T]` matrices |

`N` packs selected positions in row-major batch/token order. `positions` maps them
back to the expanded model sequence. Layer indices and algorithm provenance are
in `metadata`. Tensor outputs are detached before returning and saved with JSON
metadata; models and processors are not serialized.
