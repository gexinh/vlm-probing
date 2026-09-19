"""Shared contracts for attention observations and tensor interventions."""

import torch
from torch import Tensor

from ..core.base import BaseMethod


class BaseAttention(BaseMethod):
    """Attention family base; subclasses implement ``run``.

    Attention uses [batch, query_head, query, key]. Masks are per-example
    boolean tensors, with True denoting a valid token. A returned edited
    attention tensor must replace the probabilities used by ``A @ V`` through
    the model adapter; editing a model's diagnostic output alone is ineffective.
    """

    family = "attention"

    @staticmethod
    def _floating(value: Tensor, name: str, ndim: int) -> None:
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")
        if value.ndim != ndim or any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must have {ndim} nonempty dimensions")

    @staticmethod
    def _mask(mask: Tensor | None, shape: tuple[int, int], reference: Tensor,
              name: str) -> Tensor:
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=reference.device)
        if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
                or tuple(mask.shape) != shape or mask.device != reference.device):
            raise ValueError(f"{name} must be boolean {shape} on {reference.device}")
        return mask

    @classmethod
    def _probabilities(cls, attention: Tensor, query_mask: Tensor | None = None,
                       key_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        cls._floating(attention, "attention", 4)
        if not torch.isfinite(attention).all() or (attention < 0).any():
            raise ValueError("attention probabilities must be finite and nonnegative")
        batch, _, queries, keys = attention.shape
        qmask = cls._mask(query_mask, (batch, queries), attention, "query_mask")
        kmask = cls._mask(key_mask, (batch, keys), attention, "key_mask")
        active = qmask[:, None, :].expand(attention.shape[:-1])
        sums = attention.sum(-1)
        tolerance = 1e-2 if attention.dtype in (torch.float16, torch.bfloat16) else 1e-5
        if not torch.allclose(sums[active], torch.ones_like(sums[active]),
                              atol=tolerance, rtol=tolerance):
            raise ValueError("each valid attention row must sum to one")
        forbidden = qmask[:, None, :, None] & ~kmask[:, None, None, :]
        if (attention.masked_select(forbidden) != 0).any():
            raise ValueError("attention assigns probability to an invalid key")
        return qmask, kmask

    @classmethod
    def _stack(cls, attention: Tensor, valid_tokens: Tensor | None) -> Tensor:
        cls._floating(attention, "attention stack [layer,batch,head,query,key]", 5)
        if attention.shape[-2] != attention.shape[-1]:
            raise ValueError("cross-layer propagation requires square self-attention")
        mask = cls._mask(valid_tokens, (attention.shape[1], attention.shape[-1]),
                         attention, "valid_tokens")
        if not mask.any(-1).all():
            raise ValueError("each example needs at least one valid token")
        for layer in attention:
            cls._probabilities(layer, mask, mask)
        return mask
