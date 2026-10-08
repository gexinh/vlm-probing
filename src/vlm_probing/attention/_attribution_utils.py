"""Validation shared by attention attribution tensor kernels."""

import torch
from torch import Tensor

from .base import BaseAttention


def matching_gradient(reference: Tensor, gradient: Tensor, name: str) -> None:
    BaseAttention._floating(gradient, name, reference.ndim)
    if (gradient.shape != reference.shape or gradient.device != reference.device
            or gradient.dtype != reference.dtype):
        raise ValueError(f"{name} must match attention shape, device, and dtype")
    if not torch.isfinite(gradient).all():
        raise ValueError(f"{name} must be finite")


def integration_metadata(steps: int | None, quadrature: str) -> dict:
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int)
                              or steps < 1):
        raise ValueError("steps must be a positive integer or None")
    if quadrature not in {"right", "left", "endpoints", "trapezoid", "provided"}:
        raise ValueError("unknown integration quadrature")
    if quadrature == "endpoints" and steps == 1:
        raise ValueError("endpoints integration requires at least two steps")
    return {"integration_steps": steps, "quadrature": quadrature,
            "gradient_input": "already_integrated_average"}


def input_feedback(attention: Tensor, gradient: Tensor, mask: Tensor) -> Tensor:
    """TAM/Chen feedback: average the integral, rectify, then average heads."""
    matching_gradient(attention[-1], gradient, "integrated_gradients")
    pair_mask = mask[:, :, None] & mask[:, None, :]
    return gradient.clamp_min(0).mean(1).masked_fill(~pair_mask, 0)


def start_index(start_layer: int, layers: int) -> None:
    if (isinstance(start_layer, bool) or not isinstance(start_layer, int)
            or not 0 <= start_layer < layers):
        raise ValueError("start_layer must index the attention stack")


def finite_result(value: Tensor) -> None:
    if not torch.isfinite(value).all():
        raise ValueError("attribution overflowed; use a wider dtype")
