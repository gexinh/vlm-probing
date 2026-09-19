"""Attention path intervention on logits before the softmax."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseCausal, Runner, selection_mask


class AttentionKnockout(BaseCausal):
    """Block selected query-key paths, preserving existing negative infinity masks.

    Input has shape ``[..., queries, keys]``. ``blocked`` is a broadcastable
    boolean mask. A runner must consume the edited *logits* and recompute softmax
    and value aggregation for an actual attention intervention.
    """

    name = "attention_knockout"

    def run(
        self,
        logits: Tensor,
        *,
        blocked: Tensor,
        runner: Runner | None = None,
    ) -> ProbeResult:
        if not isinstance(logits, Tensor) or not logits.is_floating_point():
            raise TypeError("logits must be a floating-point torch.Tensor")
        if logits.ndim < 2 or logits.shape[-1] == 0:
            raise ValueError("logits must have nonempty query and key axes")
        if logits.shape[-2] == 0:
            raise ValueError("logits must have nonempty query and key axes")
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise ValueError("logits may contain finite values or negative infinity only")
        selected = selection_mask(logits, blocked)
        edited = logits.masked_fill(selected, float("-inf"))
        if not torch.isfinite(edited).any(dim=-1).all():
            raise ValueError("knockout leaves at least one query row with no legal key")
        return self._intervention_result(
            logits, edited, selected, runner=runner,
            metadata={"stage": "pre_softmax", "softmax_dimension": -1},
            extra_tensors={"probabilities": edited.softmax(dim=-1)},
        )
