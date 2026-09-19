# Model adapters

The adapter owns model-specific execution. Algorithm classes remain independent
of architecture paths. A public adapter supplies `model`, `model_id`, `readout`,
`spec`, and `run`; `TorchModelAdapter` implements scoped module hooks and is the
usual starting point.

## An explicit custom adapter

This complete binding uses the included `TinyModel`. A real model needs its own
verified module paths and sequence layout provider.

```python
import torch
from examples.tiny_model import TinyModel
from vlm_probing import (
    HookPoint, ModelReadout, ModelSpec, Prober, TokenLayout, TorchModelAdapter,
)

model = TinyModel().eval()

def layout(inputs):
    ids = inputs["input_ids"]
    n_visual = inputs["image_tokens"].shape[1]
    image_ids = torch.zeros(ids.shape[0], n_visual, dtype=torch.long, device=ids.device)
    expanded_ids = torch.cat([image_ids, ids], dim=1)
    valid = torch.ones_like(expanded_ids, dtype=torch.bool)
    visual = torch.zeros_like(valid)
    visual[:, :n_visual] = True
    return TokenLayout(valid, visual, token_ids=expanded_ids)

adapter = TorchModelAdapter(
    model,
    sites={
        "layer0": HookPoint("layers.0"),
        "layer1": HookPoint("layers.1"),
        "embedding_space": HookPoint("layers.0", kind="input", selector=0),
    },
    model_id="my-checkpoint-revision",
    readout=ModelReadout(model.lm_head, norm=model.norm),
    spec=ModelSpec(
        residuals={0: "layer0", 1: "layer1"},
        embeddings="embedding_space",
        input_embeddings=model.embedding.weight,
        layout=layout,
    ),
)
probe = Prober(model, adapter=adapter)
print(probe.describe())
```

This adapter supports residual lenses/interventions and EmbedLens. It deliberately
does not claim attention or head access. The full demonstration contract is in
[`TinyModel.probing_adapter`](../examples/tiny_model.py).

## Hook and sequence contracts

- `HookPoint(module, kind="output", selector=None)` captures a tensor output.
  An integer/string selector selects a tuple item/dictionary value.
- `kind="input"` needs a selector: an integer positional argument index or a
  keyword name. Confirm which calling convention the real model uses.
- Each requested site must execute exactly once per forward. Repeated/shared
  sites and missing sites raise errors. Hooks are removed on success and failure.
- An intervention must preserve shape, dtype, and device. It must replace the
  tensor actually consumed by downstream computation.
- The adapter runs in evaluation mode and restores every module's prior training
  flag. Gradient captures retain the real graph; downstream in-place mutations
  are rejected. It does not update parameters or their accumulated gradients.
- Full-sequence, uncached inputs are required. HF adapter defaults supply
  `use_cache=False` and `return_dict=True`; custom models declare their own
  `forward_defaults`. Explicit caches are rejected.
- `ModelSpec.layout(kwargs)` returns masks in expanded sequence coordinates. The
  layout is validated against actual captured/output sequence lengths. For a
  particular batch, `ProbeInputs(kwargs, layout=...)` overrides the provider.
- `valid` is required; `visual` is required for visual/text selection. With no
  `prompt` mask, all valid positions are considered prompt. `token_ids` enables
  strict alignment and teacher-forced scoring. The library does not guess where
  a model inserts visual tokens.

## Optional capabilities

| ModelSpec field | Contract |
| --- | --- |
| `residuals` | `{layer: site}`, raw residual block outputs, before the final readout norm |
| `attentions` | `{layer: site}`, actual probabilities `[B,H,Q,K]`; observational unless declared editable |
| `attention_scores` | `{layer: site}`, editable masked logits before softmax |
| `editable_attention` | Layer indices whose probability site is consumed by `A @ V` after the hook |
| `heads` | `{layer: site}`, projected heads `[B,T,H,D_residual]`, or raw inputs for `project_heads` |
| `project_heads(layer, tensor)` | Convert a head capture to post-output-projection residual contributions |
| `embeddings` | Site aligned to the expanded sequence in input embedding space |
| `input_embeddings` | Input embedding table `[vocabulary, residual_dimension]` |
| `linear_readout(full_residual)` | Return `(weight[V,D], inverse_norm_scale[B,T,1], center_bool)` |
| `edges` | `{edge_name: site}` for independently replaceable actual graph messages |
| `alignment_keys` | Model-kwargs tensor keys compared before paired interventions |
| `forward_defaults` | Explicit default model kwargs merged before public execution |

Head projection must split the *query-head* axis correctly, including GQA. Exclude
output-projection biases from individual head contributions. Fixed-scale logit
attribution uses the full residual's denominator; fold norm gain into the readout
weight and handle biases separately. LayerNorm needs centering; RMSNorm does not.

A returned attention diagnostic is not necessarily editable. The native HF
adapters expose eager attention observations but do not mark them editable. No
fused kernels or global Transformers classes are monkey-patched.

For EAP-IG, every edge must be an independently replaceable message at a consumer.
A list of residual nodes does not establish an edge-level circuit. Joint replay
must actually use all supplied messages; identical aliases are rejected by the
hook adapter. The current public implementation stacks equally shaped edge tensors.

## Registering an architecture

```python
from vlm_probing import register_adapter

register_adapter(MyModelClass, make_adapter)
probe = Prober(my_model)  # make_adapter(my_model)
```

Registration matches the exact Python class, so unknown subclasses are not silently
assumed compatible. Alternatively implement `model.probing_adapter()` returning
a configured adapter. Registration takes precedence over that protocol and the
built-in HF adapters. `replace=True` explicitly replaces an existing registration.

Built-in HF adapters are version-gated to Transformers 4.57.x and class-gated to
LlamaForCausalLM and Qwen2_5_VLForConditionalGeneration. Their contracts follow the
[official Llama implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py)
and [official Qwen2.5-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py).
Use an explicit adapter for other versions/architectures; update tests before
extending this list.

## Integration checks

Before declaring a model supported, verify final-layer readout equals model logits,
self-patching leaves outputs unchanged, nontrivial edits change downstream scores,
layout masks respect padding/visual expansion, and head projections sum to the
bias-free attention output. Check hook cleanup and model-state restoration after
exceptions. Gradient probes need graph-preserving sites even when model parameters
are frozen. Multi-device and quantized execution require separate validation.
