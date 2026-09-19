"""Explicit Hugging Face contracts, tested against Transformers 5.3.0.

Only native Transformers classes are registered. Returned attention weights are
observational: changing them cannot replace probabilities already used by A @ V.
"""
from importlib.metadata import version

import torch

from .spec import CapabilityError, ModelSpec, TokenLayout
from .torch import HookPoint, ModelReadout, TorchModelAdapter


# Exact module/class identities; wrapper structure is never guessed from shapes.
MODEL_CONTRACTS = {
    (f"transformers.models.{module}.modeling_{module}", name): (prefix, visual_ids)
    for module, name, prefix, visual_ids in (
        ("llama", "LlamaForCausalLM", "model", ()),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen3_vl", "Qwen3VLForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration", "model.language_model", ("image_token_id", "video_token_id")),
        ("idefics3", "Idefics3ForConditionalGeneration", "model.text_model", ("image_token_id",)),
        ("internvl", "InternVLForConditionalGeneration", "model.language_model", ("image_token_id",)),
        ("llava", "LlavaForConditionalGeneration", "model.language_model", ("image_token_index",)),
    )
}


def make_adapter(model):
    installed = version("transformers")
    release = installed.split(".")[:2]
    name = type(model).__name__
    legacy = release == ["4", "57"] and name in {
        "LlamaForCausalLM", "Qwen2_5_VLForConditionalGeneration"
    }
    if release != ["5", "3"] and not legacy:
        raise CapabilityError(f"HF adapters require transformers 5.3.x (tested: 5.3.0), got {installed}; "
                              "install vlm-probing[transformers] or pass an explicit adapter")
    prefix, visual_fields = MODEL_CONTRACTS[(type(model).__module__, name)]
    decoder = model.get_submodule(prefix)
    # Composite architectures allow other text backbones; only audited ones count.
    allowed = {"Idefics3ForConditionalGeneration": {"llama"},
               "InternVLForConditionalGeneration": {"qwen2", "llama"},
               "LlavaForConditionalGeneration": {"llama"}}
    if name in allowed and decoder.config.model_type not in allowed[name]:
        raise CapabilityError(f"{name} text backbone {decoder.config.model_type!r} is not supported")
    hybrid = name == "Qwen3_5ForConditionalGeneration"
    sites, residuals, heads, attentions = {}, {}, {}, {}
    for i, layer in enumerate(decoder.layers):
        residuals[i] = f"residual.{i}"
        # Include Qwen3-VL's DeepStack addition after the block. This also avoids
        # relying on decoder-layer output containers that differ by HF version.
        following = f"{prefix}.layers.{i + 1}" if i + 1 < len(decoder.layers) else f"{prefix}.norm"
        sites[residuals[i]] = HookPoint(following, kind="input", selector=0)
        if hybrid and layer.layer_type != "full_attention":
            continue
        heads[i] = f"heads.{i}"
        sites[heads[i]] = HookPoint(f"{prefix}.layers.{i}.self_attn.o_proj", kind="input", selector=0)
        if layer.self_attn.config._attn_implementation == "eager":
            attentions[i] = f"attention.{i}"
            sites[attentions[i]] = HookPoint(f"{prefix}.layers.{i}.self_attn", selector=1)
    sites["embeddings"] = HookPoint(f"{prefix}.layers.0", kind="input", selector=0)

    def layout(inputs):
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2:
            raise CapabilityError("HF auto layout requires input_ids; supply ProbeInputs(layout=...) otherwise")
        mask = inputs.get("attention_mask")
        valid = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != ids.shape:
            raise CapabilityError("HF auto layout requires a 2D attention_mask aligned with input_ids")
        visual = torch.zeros_like(ids, dtype=torch.bool)
        for field in visual_fields:
            visual |= ids == getattr(model.config, field)
        return TokenLayout(valid, visual & valid, token_ids=ids)

    def project_heads(i, hidden):
        attn = decoder.layers[i].self_attn
        count = attn.config.num_attention_heads
        # H * head_dim can differ from residual width. Qwen3.5 applies its output
        # gate before o_proj, so this site captures the actual gated values.
        values = hidden.reshape(*hidden.shape[:2], count, attn.head_dim)
        weights = attn.o_proj.weight.reshape(attn.o_proj.out_features, count, attn.head_dim)
        return torch.einsum("bthd,ohd->btho", values, weights)

    def linear_readout(full):
        norm = decoder.norm
        norm_weight = 1.0 + norm.weight if hybrid else norm.weight
        eps = norm.eps if hybrid else norm.variance_epsilon
        weight = model.lm_head.weight * norm_weight
        scale = (full.float().square().mean(-1, keepdim=True) + eps).rsqrt()
        return weight.to(full), scale.to(full), False

    notes = [
        "Residual layer i is measured after block i and visual injection, before the next block/final norm.",
        "Attention observation requires eager attention; projected heads exclude output bias.",
        "Native HF probability/logit intervention sites and computational edges are unavailable.",
        f"Uncached full-sequence image/text forwards; Transformers {installed}.",
    ]
    if hybrid:
        notes.append("Qwen3.5 head/attention methods cover full_attention layers only; propagation cannot skip linear layers.")
    if name == "Qwen3VLForConditionalGeneration":
        notes.append("Rollout/relevance summarize language self-attention; DeepStack injection is not an attention edge.")
    spec = ModelSpec(
        residuals, attentions=attentions, heads=heads, embeddings="embeddings",
        input_embeddings=decoder.embed_tokens.weight, layout=layout,
        project_heads=project_heads, linear_readout=linear_readout,
        alignment_keys=("image_grid_thw", "video_grid_thw", "position_ids", "second_per_grid_ts",
                        "pixel_attention_mask", "image_sizes", "token_type_ids", "mm_token_type_ids"),
        forward_defaults={"use_cache": False, "return_dict": True}, notes=tuple(notes),
    )
    model_id = getattr(model.config, "_name_or_path", "") or name
    return TorchModelAdapter(model, sites, model_id=model_id,
                             readout=ModelReadout(model.lm_head, norm=decoder.norm), spec=spec)
