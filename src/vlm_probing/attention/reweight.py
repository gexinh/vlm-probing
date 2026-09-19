"""Post-softmax attention scaling, with an explicit normalization choice."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseAttention


class AttentionReweight(BaseAttention):
    """Multiply probabilities by nonnegative weights before value aggregation.

    ``renormalize=False`` changes total attention mass and is a different
    intervention from probability redistribution. The caller/adapter must
    install the returned tensor before A @ V; ``run`` does not execute a model.
    """

    name = "attention_reweight"

    def __init__(self, *, renormalize: bool = True):
        self.renormalize = renormalize

    def run(self, attention: Tensor, weights: Tensor | float, *,
            query_mask: Tensor | None = None,
            key_mask: Tensor | None = None) -> ProbeResult:
        qmask, _ = self._probabilities(attention, query_mask, key_mask)
        weights = torch.as_tensor(weights, dtype=attention.dtype, device=attention.device)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("attention weights must be finite and nonnegative")
        try:
            shape = torch.broadcast_shapes(attention.shape, weights.shape)
        except RuntimeError as error:
            raise ValueError("weights are not broadcastable to attention") from error
        if shape != attention.shape:
            raise ValueError("weights must not expand the attention shape")
        weighted = attention * weights
        active = qmask[:, None, :, None]
        mass = weighted.sum(-1, keepdim=True)
        if not torch.isfinite(weighted).all() or not torch.isfinite(mass).all():
            raise ValueError("attention reweighting overflowed; use a wider dtype")
        if (mass.masked_select(active) <= 0).any():
            raise ValueError("reweighting removed every key of a valid query")
        if self.renormalize:
            weighted = weighted / mass.masked_fill(~active, 1)
        edited = torch.where(active, weighted, attention)
        return ProbeResult(method="attention_reweight",
                           tensors={"attention": edited, "row_mass": edited.sum(-1)},
                           metadata={"renormalize": self.renormalize,
                                     "site": "post_softmax_before_value_aggregation"})
