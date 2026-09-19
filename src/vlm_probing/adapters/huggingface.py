"""Adapters for the explicitly tested Transformers 4.57.x model contracts.

    https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models
    Returned attention weights are observational only: editing them cannot
    replace the probabilities already used by the attention implementation.
"""
from importlib.metadata import version

import torch

from .spec import CapabilityError, ModelSpec, TokenLayout
from .torch import HookPoint, ModelReadout, TorchModelAdapter


def make_adapter(model):
    installed = version("transformers")
    if installed.split(".")[:2] != ["4", "57"]:
        raise CapabilityError(f"Automatic HF adapters require transformers 4.57.x, got {installed}; "
                              "use an explicit adapter for other versions")
    multimodal = type(model).__name__ == "Qwen2_5_VLForConditionalGeneration"
    prefix = "model.language_model" if multimodal else "model"
    decoder = model.get_submodule(prefix)
    sites, residuals, heads, attentions = {}, {}, {}, {}
    for i, layer in enumerate(decoder.layers):
        residuals[i], heads[i] = f"residual.{i}", f"heads.{i}"
        # 4.57 Llama layers return Tensor; Qwen2.5-VL layers return tuple.
        sites[residuals[i]] = HookPoint(f"{prefix}.layers.{i}", selector=0 if multimodal else None)
        sites[heads[i]] = HookPoint(f"{prefix}.layers.{i}.self_attn.o_proj", kind="input", selector=0)
        if layer.self_attn.config._attn_implementation == "eager":
            attentions[i] = f"attention.{i}"
            sites[attentions[i]] = HookPoint(f"{prefix}.layers.{i}.self_attn", selector=1)
    sites["embeddings"] = HookPoint(f"{prefix}.layers.0", kind="input", selector=0)

    def layout(inputs):
        ids = inputs.get("input_ids")
        if ids is None or ids.ndim != 2:
            raise CapabilityError("HF auto layout requires input_ids; supply ProbeInputs(layout=...) otherwise")
        valid = inputs.get("attention_mask", torch.ones_like(ids)).bool()
        visual = torch.zeros_like(ids, dtype=torch.bool)
        if multimodal:
            visual = (ids == model.config.image_token_id) | (ids == model.config.video_token_id)
        return TokenLayout(valid, visual & valid, token_ids=ids)

    def project_heads(i, hidden):
        attn = decoder.layers[i].self_attn
        count = attn.config.num_attention_heads
        values = hidden.reshape(*hidden.shape[:2], count, attn.head_dim)
        weights = attn.o_proj.weight.reshape(attn.o_proj.out_features, count, attn.head_dim)
        return torch.einsum("bthd,ohd->btho", values, weights)

    def linear_readout(full):
        norm = decoder.norm
        weight = model.lm_head.weight * norm.weight
        scale = (full.float().square().mean(-1, keepdim=True) + norm.variance_epsilon).rsqrt()
        return weight.to(full), scale.to(full), False

    spec = ModelSpec(
        residuals, attentions=attentions, heads=heads, embeddings="embeddings",
        input_embeddings=decoder.embed_tokens.weight, layout=layout,
        project_heads=project_heads, linear_readout=linear_readout,
        alignment_keys=("image_grid_thw", "video_grid_thw", "position_ids", "second_per_grid_ts"),
        forward_defaults={"use_cache": False, "return_dict": True},
        notes=("Attention observation requires eager attention at model construction.",
               "Native HF probability/logit intervention sites and computational edges are unavailable.",
               "Uncached full-sequence forward only; tested with Transformers 4.57.6."),
    )
    model_id = getattr(model.config, "_name_or_path", "") or type(model).__name__
    return TorchModelAdapter(model, sites, model_id=model_id,
                             readout=ModelReadout(model.lm_head, norm=decoder.norm), spec=spec)
