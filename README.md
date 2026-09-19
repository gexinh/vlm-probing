# VLM Probing

**Read, trace, and intervene in vision-language model computations.**

VLM Probing is a small PyTorch library for studying what intermediate states
encode, how attention connects visual and language tokens, and how changes to
those computations affect an answer.

Bind a model once, configure a method, and call `.run(inputs)`:

```python
probe = Prober(model, processor=processor)
method = probe.lens.logit(layers=[8, 16, 24], tokens="last_prompt")
result = method.run(inputs)
```

| Collection | Question | Examples |
| --- | --- | --- |
| `probe.lens` | What can we decode from an internal state? | Logit, Tuned, Attention, Jacobian, EmbedLens, Patchscope |
| `probe.causal` | How does an intervention change an answer? | Patching, ablation, knockout, attribution, steering |
| `probe.attention` | Where does attention go, and what happens when it changes? | Profiles, rollout, relevance, head attribution, reweighting |

**Version 0.2.0** includes 18 tensor-level algorithms and model-bound factories for
all 18. Available methods depend on the adapter's actual capabilities. The
original tensor APIs remain available for custom experiments.

[Installation](#installation) · [Quick start](#quick-start) ·
[Models](#model-support) · [Methods](#method-catalog) ·
[API guide](docs/API.md) · [Custom adapters](docs/ADAPTERS.md)

## Installation

Python 3.10+ and PyTorch 2.2+ are required. From this checkout:

```bash
python -m pip install -e .
```

For optional Hugging Face integration:

```bash
python -m pip install -e '.[transformers]'
```

The built-in HF adapters target **Transformers 4.57.x**, tested with **4.57.6**.
Other versions need an explicit adapter. Installation does not download model
weights. These commands install the local checkout; no PyPI publication is assumed.

## Quick start

This complete example runs on CPU from the repository root. `TinyModel` is an
included two-layer decoder with real attention operations and supplied visual
embeddings. Its weights are random: this demonstrates execution, not visual
reasoning performance.

```python
import torch
from examples.tiny_model import TinyModel
from vlm_probing import Prober, TokenMargin

torch.manual_seed(7)
model = TinyModel().eval()
probe = Prober(model)
clean = {
    "input_ids": torch.tensor([[1, 2, 3]]),
    "image_tokens": torch.randn(1, 2, 6),
}
corrupt = {**clean, "image_tokens": torch.zeros_like(clean["image_tokens"])}
metric = TokenMargin(positive=4, negative=5)

# Decode the last prompt position at two layers.
decoded = probe.lens.logit(layers=[0, 1]).run(clean)
print(decoded.tensors["logits"].shape)  # [2 layers, 1 selected position, 12 words]
print(decoded.tensors["positions"])    # [[0, 4]]: batch index, expanded token index

# Independently restore each layer's visual states from clean into corrupt.
patched = probe.causal.patch(layers=[0, 1], tokens="visual").run(
    corrupt, source=clean, metric=metric,
)
print(patched.tensors["effect"])       # [layer, batch]: intervention minus baseline

# Measure visual/text attention mass.
profile = probe.attention.profile(layers=[0, 1]).run(clean)
print(profile.tensors["group_mass"].shape)  # [layer, batch, head, query, group]

decoded.save("outputs/logit_lens")
```

In causal probes, `layers=[0, 1]` means **two independent interventions**. It does
not patch both layers in the same forward pass.

Run the complete examples:

```bash
python examples/prober_quickstart.py   # all 18 public factories
python examples/hf_smoke.py            # optional HF extra; random Llama + Qwen2.5-VL
python examples/all_lenses.py          # all six lower-level tensor lens APIs
```

Without installation, prefix these commands with `PYTHONPATH=src`.

## Model support

`Prober` accepts an existing model without loading another copy or changing its
weights. Auto integration uses known model classes, a registered factory, or a
model-provided `probing_adapter()` method. Unknown models need an explicit adapter;
internal semantics are not inferred from tensor shapes.

| Model / integration | Current scope | Validation |
| --- | --- | --- |
| Included `TinyModel` | All 18 public factories | CPU forwards, gradients, interventions, calibration |
| HF `LlamaForCausalLM` | Six lenses; residual interventions; head readout; eager attention observation | Small random model, padding and GQA |
| HF `Qwen2_5_VLForConditionalGeneration` | Same, plus image/video marker layout and visual embedding capture | Small random image model with an actual vision encoder and unequal image grids |
| Other PyTorch models | Explicit adapter + `ModelSpec`, or registered factory | Depends on the supplied contract |

For HF attention observation, construct/load the model with
`attn_implementation="eager"`. Residual and head probes do not require observable
attention matrices. Recreate the Prober after changing backends or replacing modules.

**Built-in HF adapters do not expose attention editing or computational edges.**
Returned HF attention matrices are diagnostic outputs: replacing them does not
change the probabilities already used in `A @ V`. These adapters therefore reject
reweighting, temperature, knockout, and EAP-IG. Those methods work with adapters
that expose the required real computation sites.

Check support without a forward pass:

```python
summary = probe.describe()
print(summary["methods"]["causal.knockout"])
# {"available": ..., "sites": ..., "requires": ..., "missing": ...}
```

For a locally loaded Qwen2.5-VL checkpoint:

```python
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vlm_probing import Prober

checkpoint = "/path/to/Qwen2.5-VL-checkpoint"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(checkpoint)
probe = Prober(model, processor=processor)

# Use the checkpoint's chat template, including its image placeholder.
# `image` is a caller-provided PIL image.
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "What is in this image?"},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = probe.prepare(prompt=prompt, image=image, device=model.device)
result = probe.lens.logit(layers=[0, 4, 8]).run(inputs)
```

Native processor outputs may also be passed directly to `.run(...)`. `prepare`
is a thin convenience function; it does not invent chat templates or move a model.

## Calibrating a lens

Tuned and Attention Lens train only their own readout parameters. Jacobian Lens
estimates a residual-to-final-residual Jacobian through the real downstream model.
Each selected layer owns a separate fitted object.

```python
# Continue from the CPU quick start. Use separate calibration/evaluation inputs
# in real experiments; `clean` is reused here only to demonstrate the interface.
binding = {
    "model_id": "tiny-seed7",
    "tokenizer_id": "integer-demo",
    "readout_id": "tiny.norm+head",
    "calibration_id": "demo-calibration-v1",
}
lens = probe.lens.tuned(layers=[0], tokens="last_prompt", binding=binding)
lens.fit(clean, steps=20, lr=0.01)
result = lens.run(clean)
print(lens.losses[0])

lens.save("outputs/tuned")
restored = probe.lens.tuned(layers=[0], binding=binding).load("outputs/tuned")
```

`fit` takes one processed calibration batch and returns the bound method. Losses
are in `method.losses[layer]` for Tuned/Attention Lens. Repeated calls retain their
weights but restart the optimizer; Jacobian fitting replaces the prior estimate.
This is not a distributed or streaming calibration framework.

Artifacts validate model, site, readout, tokenizer, calibration identity, and
configuration. Identity strings are caller declarations, not automatic checkpoint
hashes; use meaningful revision/hash-based IDs. The public API rejects unfitted
calibrated lenses and artifact saves with unspecified identities.

## Method catalog

All factories configure reusable objects with `.run(inputs, ...)`. Calibrated
lenses additionally expose `.fit`, `.save`, and `.load`.

| Factory | Computation / preparation |
| --- | --- |
| `lens.logit(layers=..., tokens=...)` | Actual final norm and language output head |
| `lens.embed(tokens="visual")` | Cosine neighbors in **input embedding** space; `.run(..., top_k=10)` |
| `lens.tuned(layers=...)` | Per-layer affine translator fitted to model output distributions |
| `lens.attention(layers=...)` | Learned per-head vocabulary maps over projected head contributions |
| `lens.jacobian(layers=...)` | Average causal Jacobian; calibration needs multiple backward passes |
| `lens.patchscope(layers=..., source_position=..., target_layer=..., target_position=...)` | Patch into `target_inputs`, returning teacher-forced logits |
| `causal.patch(layers=..., tokens=...)` | Restore activations from `source` inputs |
| `causal.ablate(layers=..., mode=...)` | Zero, mean-reference, or resampled-reference replacement |
| `causal.knockout(layers=..., queries=..., keys=...)` | Block actual pre-softmax query-key paths |
| `causal.attribute(layers=..., tokens=...)` | First-order gradient × activation difference |
| `causal.eap_ig(edges=..., steps=...)` | Integrated gradients on declared editable edge messages |
| `causal.steer(layers=..., strength=...)` | Add a caller-provided residual `direction` |
| `attention.profile(layers=...)` | Modality mass, entropy, concentration, maximum probability |
| `attention.head_logits(layers=...)` | Fixed-scale direct head contribution; no learned decoder |
| `attention.rollout(layers=...)` | Residual-aware propagation through consecutive attention layers |
| `attention.relevance(layers=...)` | Positive-gradient self-attention propagation for a metric |
| `attention.reweight(layers=..., weight=...)` | Reweight probabilities before value aggregation |
| `attention.temperature(layers=..., temperature=...)` | Scale actual attention logits, preserving masks |

The [API guide](docs/API.md) documents call patterns and output shapes.
[Method semantics](docs/METHODS.md) documents the lower-level computations.

## Tokens, scores, and results

- Select tokens using `"all"`, `"visual"`, `"text"`, `"last_prompt"`, an integer,
  a list of positions, or a boolean `[batch, expanded_sequence]` mask. Coordinates
  refer to the sequence **after visual-token expansion**. Padding is excluded.
- `"last_prompt"` means the last valid token unless a `TokenLayout.prompt` mask is
  supplied. For inputs containing teacher-forced answers, provide that mask when
  using this selector. `SequenceLogProb` instead uses an explicit answer mask and
  aligned token IDs to score a multi-token answer with the correct one-token shift.
- Paired interventions check valid/visual/prompt masks, token IDs, and declared
  grid/position metadata. `alignment="position"` permits different token IDs but
  does not bypass layout/grid checks. Semantic image correspondence remains an
  experimental assumption.
- Lens logits have shape `[layer, selected_position, vocabulary]`. A
  `positions[selected_position, 2]` tensor records batch/token indices, supporting
  unequal visual-token counts without padding the result.
- Outputs are `ProbeResult(tensors=..., metadata=...)`, detached and moved to CPU
  by default. Change their destination with `Prober(..., result_device=...)`.
  `.save(directory)` writes `tensors.pt` and `metadata.json`.
- Effects are **intervention minus baseline**. Attention magnitude and lens
  decodability alone do not establish causal importance.

## Extending the library

```text
BaseMethod
├── BaseLens       → six concrete algorithms
├── BaseCausal     → six concrete algorithms
└── BaseAttention  → six concrete algorithms

Prober
├── LensMethods       → model-bound lens execution
├── CausalMethods     → interventions and scoring
└── AttentionMethods  → attention observation and editing
```

The public collections compose the existing algorithm classes. Model hooks live
in adapters rather than individual algorithms.

```text
src/vlm_probing/
├── prober.py        # model binding, preparation, capability report
├── api/             # three collections and shared execution helpers
├── adapters/        # scoped hooks, ModelSpec, HF integration, registration
├── lenses/          # BaseLens and six algorithms
├── causal/          # BaseCausal and six algorithms
├── attention/       # BaseAttention and six algorithms
├── core/            # shared method/result types
└── metrics.py       # token margin and teacher-forced sequence scoring
```

For another architecture, provide a [model adapter](docs/ADAPTERS.md). For another
algorithm, subclass its family base and add a model-bound factory. Existing
low-level APIs remain available for custom loops:

```python
from vlm_probing.lenses import LogitLens
# readout and hidden_states are supplied by your experiment.
tensor_result = LogitLens(readout).run(hidden_states)
```

## Scope and validation

This release validates computation and integration on CPU models, not pretrained
VLM benchmark reproduction. HF validation uses randomly initialized small models,
including the actual Qwen2.5-VL image path. Pretrained checkpoints, video inference,
quantization, distributed execution, and model sharding have not been validated.
Execution uses full-sequence, uncached forwards.

- EAP-IG is the simultaneous **edge-activation-space** IG variant. It requires
  explicitly declared edges, and does not implement automatic circuit extraction
  or the upstream input-embedding interpolation path.
- Attention Relevance is positive-gradient self-attention propagation, not the
  full multimodal/encoder-decoder explainability pipeline.
- Patchscope returns teacher-forced logits; autoregressive explanation generation
  is not included.
- EmbedLens provides semantic readout, not the full paper's clustering/pruning
  experiments.
- Exact Jacobian calibration and per-head full-vocabulary decoders can be expensive.
  Start with selected layers and small batches. Pretrained lens artifacts and
  large-scale calibration jobs are not bundled.

```bash
python -m unittest discover -s tests -v
```

HF tests are skipped if the optional dependency is absent. The
[v0.2.0 report](docs/VERSION_0_2.md) records goals, changes, expected results, and
validation. The [v0.1.0 report](docs/VERSION_PLAN.md) is retained as history.

## References and related projects

Please cite the original method papers when using these methods in research.

- [Tuned Lens](https://github.com/AlignmentResearch/tuned-lens)
- [Attention Lens](https://github.com/msakarvadia/AttentionLens)
- [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
- [EmbedLens](https://github.com/EIT-NLP/EmbedLens)
- [Patchscopes](https://github.com/PAIR-code/interpretability/tree/master/patchscopes)
- [Attention Rollout](https://aclanthology.org/2020.acl-main.385/)
- [Attention Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
- [EAP-IG](https://github.com/hannamw/EAP-IG)

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[NNsight](https://github.com/ndif-team/nnsight), and Tuned Lens are useful related
libraries. Their documentation informed this README's task-first organization.
The [literature review](docs/research/04_lenses_and_additions.md) and
[paper catalog](catalog/papers.json) record the research behind method selection;
they are not lists of additional implemented features.
