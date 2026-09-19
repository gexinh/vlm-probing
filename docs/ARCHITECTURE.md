# Architecture: v0.2.0

The library separates algorithm computation from model-specific execution and
user-facing experiment configuration.

```text
user's model + processor
          │
          ▼
Prober ── LensMethods / AttentionMethods / CausalMethods
          │ configure a bound method, then run(inputs)
          ▼
adapter + ModelSpec ── capture / token layout / actual model interventions
          │
          ▼
BaseMethod → BaseLens / BaseAttention / BaseCausal → concrete tensor algorithm
          │
          ▼
ProbeResult: detached tensors + positions/layers/configuration metadata
```

## Responsibilities

- `prober.py` binds an existing model, resolves an adapter, prepares processor
  inputs, and reports capabilities without inference.
- `api/lenses.py`, `api/attention.py`, and `api/causal.py` expose the three method
  collections. Small bound executors capture inputs, call the original algorithms,
  replay actual interventions, and aggregate results. Shared scoring, alignment,
  evaluation-mode handling, and serialization metadata are in `api/common.py`.
- `adapters/spec.py` declares sequence semantics and available sites. `torch.py`
  implements scoped hooks, restores mixed train/eval flags, and rejects missing,
  repeated, or mutated captures. `auto.py` uses explicit registration/protocols;
  `huggingface.py` handles two version-gated, tested architectures.
- `lenses/`, `attention/`, and `causal/` retain their original family base classes
  and six concrete algorithms each. They remain independently usable. No second
  copy of an algorithm is embedded in the public facade.
- `metrics.py` supplies token margins and correctly shifted teacher-forced answer
  log probabilities. Public metric objects receive resolved token metadata.

The facade uses composition. This preserves the existing inheritance contracts
without adding a separate model-specific subclass for every algorithm/model pair.
Only fitted methods own calibration state and expose fit/save/load.

## Execution contracts

Lens factories select layers and tokens before execution. Captures come from a
single model forward, and selected vectors are packed with explicit batch/token
coordinates before vocabulary decoding. Fitted lenses hold separate state per
layer, with checkpoint/site/readout/tokenizer/calibration identity validation.

Causal and attention-editing sweeps perform independent per-layer experiments.
Source tensors and the receiver baseline are captured once; each edited forward
starts from the original receiver inputs. Tensor replacement alone is never
reported as a measured effect. Attribution methods retain a differentiable graph
and never accumulate gradients into the caller's model parameters.

Rollout/relevance instead combine consecutive layers in forward order. Head
readout uses explicit per-head projection and full-residual normalization rules.
A diagnostic attention output is never treated as an editable value-path input.

## Deliberate boundaries

The common interface is a full-sequence, single-model execution interface, not a
scheduler, generation backend, model loader, or distributed training system.
Model-dependent chat templates and custom layouts remain explicit. Unknown model
architectures and missing capabilities fail with actionable errors.

Built-in HF adapters were verified on small random models, including the actual
Qwen2.5-VL vision path. Pretrained checkpoints, video, quantization, sharding,
KV-cache generation, and native HF attention edits remain unvalidated/unsupported
as documented in the README. See [adapter contracts](ADAPTERS.md),
[API semantics](API.md), and [algorithm scope](METHODS.md).
