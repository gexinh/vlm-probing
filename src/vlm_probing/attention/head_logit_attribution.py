"""Linear, additive attribution of projected attention heads to vocabulary logits."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseAttention


class HeadLogitAttribution(BaseAttention):
    """Read head contributions under an explicitly fixed linear convention.

    Heads must already be mapped through their blocks of W_O into residual
    space. ``readout_weight`` is [vocab,residual_dim], with any norm gain folded
    in by the caller. ``fixed_scale`` is the *inverse* normalization denominator
    computed from the full residual, shared across heads. For LayerNorm set
    ``center=True``; for RMSNorm leave it False. Norm and projection biases are
    separate contributions and must not be added once per head.

    This is neither independently normalized head logit lens nor a causal
    head ablation. Contributions sum to the specified linear readout of the
    sum of supplied heads; other residual components are not included.
    """

    name = "head_logit_attribution"

    def run(self, projected_heads: Tensor, readout_weight: Tensor, *,
            fixed_scale: Tensor | float = 1.0, center: bool = False) -> ProbeResult:
        self._floating(projected_heads, "projected_heads [B,H,Q,D]", 4)
        self._floating(readout_weight, "readout_weight [V,D]", 2)
        if (projected_heads.shape[-1] != readout_weight.shape[-1]
                or projected_heads.device != readout_weight.device
                or projected_heads.dtype != readout_weight.dtype):
            raise ValueError("heads and readout must share feature dimension, device, and dtype")
        if not torch.isfinite(projected_heads).all() or not torch.isfinite(readout_weight).all():
            raise ValueError("head outputs and readout must be finite")
        scale = torch.as_tensor(fixed_scale, device=projected_heads.device,
                                dtype=projected_heads.dtype)
        expected = (projected_heads.shape[0], projected_heads.shape[2], 1)
        if scale.ndim != 0 and tuple(scale.shape) != expected:
            raise ValueError(f"fixed_scale must be scalar or {expected}, shared across heads")
        if not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("fixed_scale must be finite and positive")
        heads = projected_heads - projected_heads.mean(-1, keepdim=True) if center else projected_heads
        heads = heads * (scale[:, None] if scale.ndim else scale)
        logits = torch.einsum("bhqd,vd->bhqv", heads, readout_weight)
        summed = logits.sum(1)
        if not torch.isfinite(logits).all() or not torch.isfinite(summed).all():
            raise ValueError("head logit computation overflowed; use a wider dtype")
        return ProbeResult(method="head_logit_attribution",
                           tensors={"head_logits": logits, "summed_logits": summed},
                           metadata={"convention": "fixed_scale_linear_readout",
                                     "center": center, "bias_included": False,
                                     "axes": ["batch", "head", "query", "vocab"]})
