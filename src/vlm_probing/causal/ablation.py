"""Zero, mean-reference, and explicitly resampled activation ablation."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseCausal, Runner, selection_mask, validate_activation, validate_pair


class Ablation(BaseCausal):
    """Replace selected entries using an explicit control distribution.

    ``mean`` averages ``reference`` over ``mean_dims``, keeping dimensions;
    callers should provide reference data independent of the tested example.
    ``resample`` takes an already sampled, same-shaped reference activation.
    """

    name = "ablation"

    def run(
        self,
        activation: Tensor,
        *,
        mode: str = "zero",
        mask: Tensor | None = None,
        reference: Tensor | None = None,
        mean_dims: tuple[int, ...] = (0,),
        runner: Runner | None = None,
    ) -> ProbeResult:
        validate_activation(activation, "activation")
        selected = selection_mask(activation, mask)
        if mode == "zero":
            if reference is not None:
                raise ValueError("zero ablation does not use a reference")
            replacement = torch.zeros_like(activation)
        elif mode in {"mean", "resample"}:
            if reference is None:
                raise ValueError(f"{mode} ablation requires an explicit reference")
            validate_activation(reference, "reference")
            if reference.device != activation.device or reference.dtype != activation.dtype:
                raise ValueError("reference and activation must share a device and dtype")
            if mode == "resample":
                validate_pair(activation, reference)
                replacement = reference
            else:
                if reference.ndim != activation.ndim:
                    raise ValueError("mean reference must have the same rank as activation")
                if not mean_dims or any(
                    not isinstance(dim, int) or not -reference.ndim <= dim < reference.ndim
                    for dim in mean_dims
                ):
                    raise ValueError("mean_dims must contain valid reference axes")
                dims = tuple(dim % reference.ndim for dim in mean_dims)
                if len(set(dims)) != len(dims):
                    raise ValueError("mean_dims must not repeat an axis")
                replacement = reference.mean(dim=dims, keepdim=True)
                if not torch.isfinite(replacement).all():
                    raise ValueError("reference mean must be finite and nonempty")
                try:
                    replacement = torch.broadcast_to(replacement, activation.shape)
                except RuntimeError as exc:
                    raise ValueError("reference mean must broadcast to activation shape") from exc
        else:
            raise ValueError("mode must be 'zero', 'mean', or 'resample'")
        edited = torch.where(selected, replacement, activation)
        return self._intervention_result(
            activation, edited, selected, runner=runner,
            metadata={"mode": mode, "mean_dims": mean_dims if mode == "mean" else None},
        )
