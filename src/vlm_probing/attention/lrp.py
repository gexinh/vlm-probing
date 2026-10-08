"""Exact, explicitly scoped Chefer CVPR 2021 ViT relevance backend."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .._vendor.chefer2021.vit import VisionTransformer
from ..core.types import ProbeResult
from .base import BaseAttention
from .chefer import CheferTransformerAttribution


CHEFER_COMMIT = "c3e578f76b954e8528afeaaee26de3f07e3fe559"


def _hf_weights(source: nn.Module):
    """Convert only the declared Hugging Face ViT classifier architecture."""
    if (type(source).__name__ != "ViTForImageClassification"
            or type(source).__module__ != "transformers.models.vit.modeling_vit"):
        raise TypeError("source must be a native Hugging Face ViTForImageClassification")
    config = source.config
    if config.hidden_act != "gelu":
        raise ValueError("the official Chefer backend requires exact GELU activation")
    if not isinstance(source.classifier, nn.Linear):
        raise ValueError("a linear image classification head is required")
    values = source.state_dict()
    pairs = {
        "cls_token": "vit.embeddings.cls_token",
        "pos_embed": "vit.embeddings.position_embeddings",
        "patch_embed.proj.weight": "vit.embeddings.patch_embeddings.projection.weight",
        "patch_embed.proj.bias": "vit.embeddings.patch_embeddings.projection.bias",
        "norm.weight": "vit.layernorm.weight", "norm.bias": "vit.layernorm.bias",
        "head.weight": "classifier.weight", "head.bias": "classifier.bias",
    }
    weights = {name: values[origin] for name, origin in pairs.items()}
    for i in range(config.num_hidden_layers):
        original, local = f"vit.encoder.layer.{i}", f"blocks.{i}"
        for suffix in ("weight", "bias"):
            qkv = [f"{original}.attention.attention.{kind}.{suffix}" for kind in ("query", "key", "value")]
            if suffix == "weight" or config.qkv_bias:
                weights[f"{local}.attn.qkv.{suffix}"] = torch.cat([values[key] for key in qkv])
            for a, b in (("attn.proj", "attention.output.dense"),
                         ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
                         ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
                weights[f"{local}.{a}.{suffix}"] = values[f"{original}.{b}.{suffix}"]
    return config, weights


def create_chefer_vit(*, source: nn.Module | None = None,
                      checkpoint: str | None = None, **configuration) -> VisionTransformer:
    """Build the pinned author architecture; download no weights implicitly.

    ``source`` copies a native HF ViT classifier, including LayerNorm epsilon.
    ``checkpoint`` loads an already-local author/timm-style ViT state dict.
    Without either, the returned model is randomly initialized for CPU checks.
    """
    if source is not None and (checkpoint is not None or configuration):
        raise ValueError("source cannot be combined with checkpoint or configuration")
    if source is not None:
        config, weights = _hf_weights(source)
        model = VisionTransformer(
            img_size=config.image_size, patch_size=config.patch_size,
            in_chans=config.num_channels, num_classes=config.num_labels,
            embed_dim=config.hidden_size, depth=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_ratio=config.intermediate_size / config.hidden_size,
            qkv_bias=config.qkv_bias, drop_rate=config.hidden_dropout_prob,
            attn_drop_rate=config.attention_probs_dropout_prob)
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                module.eps = config.layer_norm_eps
        parameter = next(source.parameters())
        model.to(device=parameter.device, dtype=parameter.dtype)
        model.load_state_dict(weights, strict=True)
        model._vlm_probing_source = getattr(config, "_name_or_path", "HF ViT classifier")
    else:
        # Match the author's ViT-base factory, rather than its generic class's
        # bias-free default, so a local official checkpoint loads directly.
        model = VisionTransformer(**{"qkv_bias": True, **configuration})
        if checkpoint is not None:
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(weights, Mapping) and "model" in weights:
                weights = weights["model"]
            model.load_state_dict(weights, strict=True)
        model._vlm_probing_source = checkpoint or "random initialization; validation only"
    model.eval()
    return model


class CheferLRP(BaseAttention):
    """Run genuine author relprop rules on the explicit Chefer ViT backend.

    Transformer attribution is the CVPR paper's proposed method. ``full``
    propagates its relevance rules to pixels; ``last_layer`` returns its final
    attention CAM. These extra modes are not the paper's separate original-LRP
    comparison, which uses a different ``ViT_orig_LRP.py`` implementation.
    Each image is processed separately, matching the original single-image
    implementation's branch normalization and class seed.
    """

    name = "chefer_lrp"

    def __init__(self, model: VisionTransformer, *, method="transformer_attribution",
                 start_layer: int = 0, result_device="cpu"):
        if type(model) is not VisionTransformer:
            raise TypeError("CheferLRP requires create_chefer_vit(); generic HF/VLM attention is insufficient")
        if method not in {"transformer_attribution", "full", "last_layer"}:
            raise ValueError("method must be transformer_attribution, full, or last_layer")
        if type(start_layer) is not int or not 0 <= start_layer < len(model.blocks):
            raise ValueError("start_layer must select an existing layer")
        self.model, self.method, self.start_layer = model, method, start_layer
        self.result_device = torch.device(result_device)
        self._active = False

    def run(self, pixel_values: Tensor, *, target: int | Tensor | None = None) -> ProbeResult:
        self._floating(pixel_values, "pixel_values [B,C,H,W]", 4)
        if not torch.isfinite(pixel_values).all():
            raise ValueError("pixel_values must be finite")
        if torch.is_inference_mode_enabled():
            raise ValueError("LRP requires autograd; run outside torch.inference_mode()")
        if self._active:
            raise RuntimeError("concurrent or nested CheferLRP execution is unsupported")
        parameter = next(self.model.parameters())
        if pixel_values.device != parameter.device or pixel_values.dtype != parameter.dtype:
            raise ValueError("pixel_values must match the backend device and dtype")
        if (pixel_values.shape[1] != self.model.patch_embed.proj.in_channels
                or tuple(pixel_values.shape[-2:]) != tuple(self.model.patch_embed.img_size)):
            raise ValueError("pixel_values must match the backend's channels and image size")
        if type(target) is int:
            targets = torch.full((len(pixel_values),), target, device=pixel_values.device)
        elif target is None:
            targets = None
        elif (isinstance(target, Tensor) and target.dtype == torch.long
              and target.shape == (len(pixel_values),)):
            targets = target.to(pixel_values.device)
        else:
            raise ValueError("target must be an integer, a long [B] tensor, or None for top class")
        if targets is not None and ((targets < 0).any() or (targets >= self.model.num_classes).any()):
            raise ValueError("target class is outside the classifier vocabulary")
        states = [(module, module.training) for module in self.model.modules()]
        logits, maps, all_cams, all_gradients, chosen = [], [], [], [], []
        self._active = True
        try:
            self.model.eval()
            with torch.enable_grad():
                for i, image in enumerate(pixel_values):
                    # This also supports fully frozen classifier parameters.
                    image = image[None].detach().clone().requires_grad_(True)
                    output = self.model(image)
                    index = output.argmax(-1) if targets is None else targets[i:i + 1]
                    seed = torch.zeros_like(output).scatter_(1, index[:, None], 1.)
                    attentions = [block.attn.get_attn() for block in self.model.blocks]
                    # autograd.grad leaves caller-owned parameter .grad buffers unchanged.
                    torch.autograd.grad((output * seed).sum(), attentions, retain_graph=True)
                    relevance = self.model.relprop(seed, method=self.method,
                                                  start_layer=self.start_layer, alpha=1)
                    if relevance.ndim == 1:
                        relevance = relevance[None]
                    cams = torch.stack([block.attn.get_attn_cam() for block in self.model.blocks])
                    gradients = torch.stack([block.attn.get_attn_gradients() for block in self.model.blocks])
                    if self.method == "transformer_attribution":
                        # Both the upstream model and this reusable kernel must agree.
                        result = CheferTransformerAttribution(start_layer=self.start_layer).run(cams, gradients)
                        independent = result.tensors["relevance"][:, 0, 1:]
                        if not torch.allclose(independent, relevance, atol=1e-6, rtol=1e-5):
                            raise RuntimeError("aggregation disagrees with the pinned author implementation")
                    if not all(torch.isfinite(v).all() for v in (output, relevance, cams, gradients)):
                        raise ValueError("the LRP backend produced nonfinite relevance")
                    logits.append(output.detach())
                    maps.append(relevance.detach())
                    all_cams.append(cams.detach())
                    all_gradients.append(gradients.detach())
                    chosen.append(index.detach())
        finally:
            for module, state in states:
                module.training = state
            self._active = False
        tensors = {"logits": torch.cat(logits), "target": torch.cat(chosen),
                   "relevance": torch.cat(maps),
                   "attention_relevance": torch.cat(all_cams, dim=1),
                   "attention_gradients": torch.cat(all_gradients, dim=1)}
        if self.method != "full":
            grid = tuple(image // patch for image, patch in zip(
                self.model.patch_embed.img_size, self.model.patch_embed.patch_size))
            tensors["patch_relevance"] = tensors["relevance"].reshape(len(pixel_values), *grid)
        return ProbeResult(f"chefer_{self.method}",
                           {key: value.to(self.result_device) for key, value in tensors.items()},
                           {"paper": "https://arxiv.org/abs/2012.09838",
                            "official_code": "https://github.com/hila-chefer/Transformer-Explainability",
                            "source_commit": CHEFER_COMMIT, "backend": "explicit_author_ViT",
                            "weights": str(self.model._vlm_probing_source),
                            "method": self.method, "alpha": 1, "start_layer": self.start_layer,
                            "single_image_relprop": True, "target": "top_class" if target is None else "explicit_class",
                            "scope": "vision classification; native LLM/VLM full DTD is unsupported"})
