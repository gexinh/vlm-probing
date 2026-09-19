"""Activation steering with an optional token-vector norm constraint."""

import math

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseCausal, Runner, selection_mask, validate_activation


class Steering(BaseCausal):
    """Add a broadcastable direction to selected entries.

    With ``preserve_norm=True``, the entire vector on the last axis is rescaled
    to its original norm. In that mode masks must select whole vectors, because
    a partial feature mask and whole-vector norm preservation are incompatible.
    Cancelling a nonzero vector exactly has no norm-preserving direction and
    raises an error. Originally zero vectors remain zero under preservation.
    """

    name = "steering"

    def run(
        self,
        activation: Tensor,
        direction: Tensor,
        *,
        strength: float = 1.0,
        mask: Tensor | None = None,
        preserve_norm: bool = False,
        runner: Runner | None = None,
    ) -> ProbeResult:
        validate_activation(activation, "activation")
        validate_activation(direction, "direction")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite")
        if direction.device != activation.device or direction.dtype != activation.dtype:
            raise ValueError("direction and activation must share a device and dtype")
        try:
            direction = torch.broadcast_to(direction, activation.shape)
        except RuntimeError as exc:
            raise ValueError("direction must broadcast to activation shape") from exc
        selected = selection_mask(activation, mask)
        edited = torch.where(selected, activation + strength * direction, activation)
        if preserve_norm:
            if activation.ndim == 0 or activation.shape[-1] == 0:
                raise ValueError("norm preservation requires a nonempty final vector axis")
            if not torch.equal(selected, selected[..., :1].expand_as(selected)):
                raise ValueError("norm-preserving steering requires whole-vector masks")
            original_norm = torch.linalg.vector_norm(activation, dim=-1, keepdim=True)
            edited_norm = torch.linalg.vector_norm(edited, dim=-1, keepdim=True)
            if ((original_norm > 0) & (edited_norm == 0)).any():
                raise ValueError("steering cancelled a nonzero vector; cannot restore its norm")
            denominator = torch.where(edited_norm > 0, edited_norm, torch.ones_like(edited_norm))
            restored = edited * (original_norm / denominator)
            edited = torch.where(selected, restored, activation)
        if not torch.isfinite(edited).all():
            raise ValueError("steering produced nonfinite activations")
        return self._intervention_result(
            activation, edited, selected, runner=runner,
            metadata={"strength": float(strength), "preserve_norm": preserve_norm},
        )
