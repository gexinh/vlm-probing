"""Native HF ViT classification: patch layouts and consumed attention taps.

The adapter preserves [B,C] classifier outputs. It deliberately exposes no
language readout, vocabulary embeddings, or text-generation capabilities.
"""
from functools import wraps
from importlib.metadata import version

import torch
from torch import nn

from .spec import CapabilityError, ModelSpec, TokenLayout
from .torch import HookPoint, TorchModelAdapter


def _install(attention):
    if hasattr(attention, "_probe_probs"):
        return
    from transformers.models.vit.modeling_vit import ALL_ATTENTION_FUNCTIONS

    def tapped(module, query, key, value, attention_mask, scaling=None, dropout=0., **kwargs):
        scale = query.size(-1) ** -.5 if scaling is None else scaling
        scores = query @ key.transpose(2, 3) * scale
        if attention_mask is not None:
            scores = scores + attention_mask
        weights = module._probe_scores(scores).softmax(-1)
        weights = nn.functional.dropout(weights, p=dropout, training=module.training)
        weights = module._probe_probs(weights)
        return (weights @ value).transpose(1, 2).contiguous(), weights

    backend = "vlm_probing_vit_eager"
    ALL_ATTENTION_FUNCTIONS.register(backend, tapped)
    attention.add_module("_probe_scores", nn.Identity())
    attention.add_module("_probe_probs", nn.Identity())
    original = attention.forward

    @wraps(original)
    def forward(*args, **kwargs):
        previous = attention.config._attn_implementation
        if previous != "eager":
            raise ValueError("ViT attention taps require attn_implementation='eager'")
        try:
            attention.config._attn_implementation = backend
            return original(*args, **kwargs)
        finally:
            attention.config._attn_implementation = previous

    attention.forward = forward


def make_adapter(model):
    if version("transformers").split(".")[:2] != ["5", "3"]:
        raise CapabilityError("native ViT adapter is audited for transformers 5.3.x")
    if model.config._attn_implementation != "eager":
        raise CapabilityError("native ViT adapter requires attn_implementation='eager'")
    layers = model.vit.encoder.layer
    sites, residuals, attentions, scores, inputs, values = {}, {}, {}, {}, {}, {}
    for i, layer in enumerate(layers):
        prefix = f"vit.encoder.layer.{i}"
        _install(layer.attention.attention)
        residuals[i] = f"residual.{i}"
        following = f"vit.encoder.layer.{i+1}" if i+1 < len(layers) else "vit.layernorm"
        sites[residuals[i]] = HookPoint(following, kind="input", selector=0)
        attentions[i], scores[i] = f"attention.{i}", f"scores.{i}"
        sites[attentions[i]] = HookPoint(f"{prefix}.attention.attention._probe_probs")
        sites[scores[i]] = HookPoint(f"{prefix}.attention.attention._probe_scores")
        inputs[i], values[i] = residuals[i-1] if i else "attention_input.0", f"values.{i}"
        if not i:
            sites[inputs[i]] = HookPoint(prefix, kind="input", selector=0)
        sites[values[i]] = HookPoint(f"{prefix}.attention.attention.value")
    sites["embeddings"] = HookPoint("vit.embeddings")

    def layout(kwargs):
        pixels = kwargs.get("pixel_values")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise CapabilityError("ViT requires pixel_values[B,C,H,W]")
        if kwargs.get("interpolate_pos_encoding", False):
            raise CapabilityError("interpolated ViT grids require an explicit custom adapter")
        patch = model.vit.embeddings.patch_embeddings
        expected = tuple(patch.image_size)
        if tuple(pixels.shape[-2:]) != expected:
            raise ValueError(f"processor must resize images to {expected}")
        valid = torch.ones(pixels.shape[0], patch.num_patches+1,
                           dtype=torch.bool, device=pixels.device)
        visual = valid.clone()
        visual[:, 0] = False
        return TokenLayout(valid, visual)

    def project_values(i, raw):
        # Concatenated V already follows the native head order. W_O bias is excluded.
        return nn.functional.linear(raw, layers[i].attention.output.dense.weight)

    spec = ModelSpec(residuals, attentions=attentions, attention_scores=scores,
                     embeddings="embeddings", layout=layout, output_kind="classification",
                     editable_attention=set(attentions), attention_inputs=inputs,
                     attention_values=values, project_values=project_values,
                     forward_defaults={"return_dict": True}, notes=(
                         "Native ViT CLS class logits are [batch,classes]; use ClassScore.",
                         "Visual tokens are row-major image patches after CLS; no vocabulary readout.",
                         "Consumed probabilities and scores support actual attention interventions.",
                         "Chefer DTD/LRP uses a separate original-rule model conversion, not native gradients.",))
    return TorchModelAdapter(model, sites, spec=spec,
                             model_id=getattr(model.config, "_name_or_path", "") or type(model).__name__)
