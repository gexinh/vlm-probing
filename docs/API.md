# Common API

[Home](../README.md) / [Documentation](README.md)

Model-specific setup lives in the [model guides](models/README.md). Individual
factory parameters and examples live in the [method guides](methods/README.md).

## Bind and inspect

```python
from vlm_probing import Prober

probe = Prober(model, processor=processor, adapter="auto", result_device="cpu")
support = probe.describe()
print(support["methods"]["lens.logit"])
# {"available": True, "sites": [...], "requires": ..., "missing": None}
```

`model` is a caller-owned model instance. Automatic selection uses exact known
classes, registered factories, or a model's `probing_adapter()` protocol.
For another architecture, pass an explicit [adapter](ADAPTERS.md).
`describe()` runs no inference and exposes method sites and missing capabilities.

## Prepare inputs

```python
inputs = probe.prepare(prompt=formatted_prompt, image=pil_image, device="cuda:0")
result = probe.lens.logit(layers=[0]).run(inputs)

# Existing processor output works directly.
native_inputs = processor(text=formatted_prompt, images=pil_image, return_tensors="pt")
native_inputs = {k: v.to("cuda:0") for k, v in native_inputs.items()}
result = probe.lens.logit(layers=[0]).run(native_inputs)
```

`prepare(*, prompt, image=None, layout=None, device=None, **kwargs)` forwards
processor arguments and returns `ProbeInputs`. Apply the model's chat template
first, as shown on each model page. The optional layout uses expanded sequence
coordinates. Native dictionary values need not all be tensors; `prepare` moves
tensor values only. Probing requires full-sequence inputs and rejects KV caches,
`use_cache=True`, and nonzero `logits_to_keep`.

## Select layers and tokens

Factory configuration is reusable: `method = probe.lens.logit(...)`, followed
by `method.run(batch)` for each batch. Layers are zero-based. For native HF
adapters, residual layer `i` is the next block's input (or final norm input),
including any visual injection after block `i`.

| Selector | Meaning |
| --- | --- |
| `layers=None` | All available sites for that method; propagation uses the contiguous prefix from layer 0. |
| `layers=3` or `layers=[3, 7]` | Explicit unique layer indices, preserving order. |
| `tokens="all"` | All valid positions. |
| `tokens="visual"` / `"text"` | Positions selected by the adapter's visual mask or its valid complement. |
| `tokens="last_prompt"` | Last prompt position, or last valid position if no prompt mask exists. |
| `tokens=4` / `[4, 5]` | Expanded sequence positions, valid in every batch example. |
| `tokens=boolean_mask` | Explicit `[batch, expanded_sequence]` selection, excluding padding. |

`queries` and `keys` use the same selector conventions. Residual interventions
at multiple layers are separate experiments; they do not compound edits across
layers. Rollout and relevance instead compose consecutive attention layers.

## Metrics and explicit layouts

For single-token candidate answers:

```python
from vlm_probing import TokenMargin

positive = processor.tokenizer.encode(" red", add_special_tokens=False)
negative = processor.tokenizer.encode(" blue", add_special_tokens=False)
assert len(positive) == len(negative) == 1
metric = TokenMargin(positive[0], negative[0], position="last_prompt")
result = probe.causal.patch(layers=[0]).run(corrupt, source=clean, metric=metric)
```

The score is positive-token logit minus negative-token logit. It needs exactly
one prediction position per example. For teacher-forced multi-token answers,
supply the full prompt+answer sequence and an answer mask:

```python
from vlm_probing import ProbeInputs, TokenLayout, SequenceLogProb

inputs = ProbeInputs(model_kwargs, TokenLayout(
    valid=valid_mask, visual=visual_mask, prompt=prompt_mask, token_ids=expanded_ids,
))
metric = SequenceLogProb(answer_mask, reduction="sum")
```

Masks are boolean `[B,T]`; token IDs are long `[B,T]`, aligned with actual model
logits. Sequence scoring shifts logits by one and excludes visual/padding
targets. Each answer token needs a preceding prediction position; EOS is scored
only when selected. Score separate complete inputs for different candidate answers.

Custom metrics may be `callable(logits) -> scalar_or_batch_tensor` or objects
with `score(logits, layout)`. Preserve the graph for gradient methods.

## Aligned interventions

`alignment="strict"` checks valid/visual/prompt masks, expanded token IDs, and
adapter-declared grid/position metadata. Source and receiver use identical
coordinates. For intentionally changed text with verified positional alignment,
`alignment="position"` relaxes only token-ID equality. General position remapping
is not implemented. Use independent reference data for ablation controls.

## Fitted-lens artifacts

Tuned, Attention, and Jacobian Lens need `fit` or `load` before `run`.
The model's parameters and accumulated gradients are preserved.

```python
binding = {
    "model_id": "checkpoint-and-revision",
    "tokenizer_id": "tokenizer-and-revision",
    "readout_id": "final-norm-and-lm-head-v1",
    "calibration_id": "dataset-split-and-preprocessing-v1",
}
lens = probe.lens.tuned(layers=[0], binding=binding)
lens.fit(calibration_inputs, steps=100, lr=1e-3)
lens.save("artifacts/tuned")
restored = probe.lens.tuned(layers=[0], binding=binding).load("artifacts/tuned")
result = restored.run(evaluation_inputs)
```

`fit` accepts one batch; `fit` and `load` return the bound object. Tuned/Attention
retain weights across repeated fits but restart optimizers; their losses are
available in `.losses[layer]`. Jacobian fitting replaces the transport estimate.
Artifact saving requires explicit identities and writes one file per layer.
Loading validates the binding and dimensions. These files use this library's
format, not a third-party lens checkpoint format.

## Results and persistence

`ProbeResult` contains `.method`, `.tensors`, and `.metadata`.
Outputs are detached and moved to `result_device` (CPU by default).

| Method | Main outputs |
| --- | --- |
| Logit, Tuned, Jacobian | `logits[L,N,V]`, `positions[N,2]`; Jacobian also `transported[L,N,D]` |
| Embed | `token_ids[1,N,K]`, `similarities[1,N,K]`, norms and positions |
| Attention Lens | `head_logits[L,N,H,V]`, summed logits and positions |
| Patchscope | Target `logits[L,B,T_target,V]` |
| Exact interventions | Baseline/intervention scores and effect: `[L,B]` or `[L]` |
| Attribution patching | `token_scores[L,B,T]`, batch-summed `estimated_effect[L]` |
| EAP-IG | `edge_scores[E]`, scalar endpoints/effect/completeness error |
| Attention profile | `entropy[L,B,H,Q]`, `group_mass[L,B,H,Q,G]` |
| Head logit attribution | `head_logits[L,B,H,Q,V]`, `summed_logits[L,B,Q,V]` |
| Rollout / relevance | Final `[B,T,T]` and intermediate `[L,B,T,T]` matrices |

`N` packs selected positions in batch/token order. Layer indices, configuration,
and provenance are recorded in metadata.

```python
result.save("outputs/probe")
tensors = torch.load("outputs/probe/tensors.pt", weights_only=True)
```

The accompanying `metadata.json` records the operation. Models and processors
are not serialized. Execution is scoped to one model forward at a time; concurrent
or nested execution on the same adapter is unsupported.
