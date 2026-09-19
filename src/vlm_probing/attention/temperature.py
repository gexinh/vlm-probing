"""Attention softmax temperature, separate from generation sampling temperature."""

import math

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseAttention


class AttentionTemperature(BaseAttention):
    """Apply scalar temperature to pre-softmax logits, preserving -inf masks.

    Returned ``attention`` must replace A in A @ V, or returned ``logits`` may
    replace the scores before the model's softmax. Input logits include the
    model's usual attention scaling and additive masks before this operation.
    All-masked inactive queries produce zero probabilities, but their returned
    logits retain -inf; adapters using logits must handle padding explicitly.
    """

    name = "attention_temperature"

    def __init__(self, temperature: float = 1.0):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)

    def run(self, logits: Tensor, *, query_mask: Tensor | None = None) -> ProbeResult:
        self._floating(logits, "attention logits [B,H,Q,K]", 4)
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError("logits may contain finite values or -inf masks only")
        qmask = self._mask(query_mask, (logits.shape[0], logits.shape[2]), logits, "query_mask")
        active = qmask[:, None, :, None]
        valid_rows = torch.isfinite(logits).any(-1)
        if (~valid_rows & qmask[:, None, :]).any():
            raise ValueError("every valid query needs at least one unmasked key")
        scaled = logits / self.temperature
        if (~torch.isfinite(scaled) & torch.isfinite(logits)).any():
            raise ValueError("temperature scaling overflowed; use a wider dtype")
        # Inactive padding rows are explicitly zero, including all-masked rows.
        attention = scaled.masked_fill(~active, 0).softmax(-1).masked_fill(~active, 0)
        return ProbeResult(method="attention_temperature",
                           tensors={"attention": attention, "logits": scaled},
                           metadata={"temperature": self.temperature,
                                     "site": "pre_softmax", "invalid_query_policy": "zero"})
