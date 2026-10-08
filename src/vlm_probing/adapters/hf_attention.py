"""Audited, weight-free consumed-attention taps for Transformers 5.3 decoders.

The numerical operations and their order match ``eager_attention_forward``.
The backend override lasts only for an attention call: decoder mask construction
must still see the ordinary eager backend. Adapter execution is non-reentrant.
"""
from functools import wraps
from importlib import import_module

import torch
from torch import nn


_AUDITED_CLASSES = {
    (f"transformers.models.{family}.modeling_{family}", name)
    for family, name in (
        ("llama", "LlamaAttention"),
        ("mistral", "MistralAttention"),
        ("qwen2", "Qwen2Attention"),
        ("qwen2_vl", "Qwen2VLAttention"),
        ("qwen2_5_vl", "Qwen2_5_VLAttention"),
        ("qwen3_vl", "Qwen3VLTextAttention"),
        ("qwen3_5", "Qwen3_5Attention"),
    )
}


def install_eager_taps(attention):
    """Return whether this exact eager architecture supports editable scores/probabilities."""
    if ((type(attention).__module__, type(attention).__name__) not in _AUDITED_CLASSES
            or attention.config._attn_implementation != "eager"):
        return False
    if hasattr(attention, "_probe_scores"):
        return True
    native = import_module(type(attention).__module__)
    functions, repeat_kv = native.ALL_ATTENTION_FUNCTIONS, native.repeat_kv

    def tapped_eager(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        scores = module._probe_scores(scores)
        weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        weights = nn.functional.dropout(weights, p=dropout, training=module.training)
        weights = module._probe_probs(weights)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return output, weights

    backend = "vlm_probing_consumed_eager"
    functions.register(backend, tapped_eager)
    attention.add_module("_probe_scores", nn.Identity())
    attention.add_module("_probe_probs", nn.Identity())
    original = attention.forward

    @wraps(original)
    def forward(*args, **kwargs):
        previous = attention.config._attn_implementation
        # Respect a caller who switches away from eager after constructing Prober.
        if previous != "eager":
            raise ValueError("editable native attention requires attn_implementation='eager'")
        try:
            attention.config._attn_implementation = backend
            return original(*args, **kwargs)
        finally:
            attention.config._attn_implementation = previous

    attention.forward = forward
    return True
