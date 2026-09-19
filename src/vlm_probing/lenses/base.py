"""Common interface and small shared utilities for tensor-based lenses."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.base import BaseMethod
from ..core.types import ProbeResult


class BaseLens(BaseMethod):
    """A lens decodes a specified representation; it does not collect it.

    Callers supply activations at a known model site. ``layer`` is provenance,
    not a request to select another tensor. Model-specific capture belongs in
    the adapter. Every concrete method documents its own tensor semantics.
    """

    family = "lens"

    @abstractmethod
    def run(self, activations: Tensor, **kwargs: Any) -> ProbeResult:
        """Return decoded tensors and enough metadata to identify the operation."""


def floating_tensor(value: Tensor, name: str, *, ndim: int | None = None) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch.Tensor")
    if (ndim is not None and value.ndim != ndim) or value.ndim == 0:
        raise ValueError(f"{name} has invalid shape {tuple(value.shape)}")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")


def decoded(readout: Callable[[Tensor], Tensor], hidden: Tensor) -> Tensor:
    logits = readout(hidden)
    floating_tensor(logits, "readout logits")
    if logits.shape[:-1] != hidden.shape[:-1]:
        raise ValueError("readout must preserve every axis except the hidden dimension")
    return logits


def position_mask(mask: Tensor | None, shape: tuple[int, ...], device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"mask must be boolean with shape {shape}")
    mask = mask.to(device)
    if not mask.any():
        raise ValueError("mask must select at least one position")
    return mask


@torch.enable_grad()
def fit_distribution(
    module: nn.Module,
    predict: Callable[[Tensor], Tensor],
    activations: Tensor,
    teacher_logits: Tensor,
    *,
    steps: int,
    lr: float,
    mask: Tensor | None,
) -> list[float]:
    """Fit only module parameters with KL(teacher || lens).

    ``autograd.grad`` avoids filling any caller-owned model parameter .grad.
    Inputs and teacher distributions are detached from their original graphs.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("lens calibration cannot run inside torch.inference_mode()")
    floating_tensor(activations, "activations")
    floating_tensor(teacher_logits, "teacher_logits")
    if not isinstance(steps, int) or steps < 1 or not 0 < lr < float("inf"):
        raise ValueError("steps must be positive and lr must be finite and positive")
    parameters = list(module.parameters())
    parameter = parameters[0]
    inputs = activations.detach().to(device=parameter.device, dtype=parameter.dtype)
    targets = teacher_logits.detach().to(device=parameter.device, dtype=torch.float32)
    selected = position_mask(mask, tuple(targets.shape[:-1]), parameter.device)
    target_log_probs = targets.log_softmax(-1)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        prediction = predict(inputs)
        if prediction.shape != targets.shape:
            raise ValueError("lens and teacher logits must have identical shapes")
        loss_by_position = F.kl_div(
            prediction.float().log_softmax(-1), target_log_probs,
            reduction="none", log_target=True,
        ).sum(-1)
        loss = loss_by_position[selected].mean()
        if not torch.isfinite(loss):
            raise ValueError("calibration loss is not finite")
        optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(loss, parameters)
        if any(not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError("calibration gradient is not finite")
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        optimizer.step()
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return losses


_BINDING_KEYS = {"model_id", "site", "readout_id", "tokenizer_id", "calibration_id"}


def save_artifact(
    path: str | Path, method: str, binding: Mapping[str, str],
    config: dict[str, Any], state: dict[str, Any],
) -> None:
    """Save weights with caller-declared checkpoint/site/calibration identity.

    IDs are declarations, not automatic hashes of a remote checkpoint. Callers
    should use immutable revision IDs or hashes where available.
    """
    missing = _BINDING_KEYS - binding.keys()
    if missing or any(not isinstance(v, str) or not v for v in binding.values()):
        raise ValueError(f"artifact binding requires nonempty strings for {sorted(_BINDING_KEYS)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "method": method, "binding": dict(binding),
                "config": config, "state": state}, path)


def load_artifact(
    path: str | Path, method: str, binding: Mapping[str, str], config: dict[str, Any],
) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1 or artifact.get("method") != method:
        raise ValueError("artifact format or method does not match this lens")
    if _BINDING_KEYS - binding.keys() or artifact.get("binding") != dict(binding):
        raise ValueError("artifact binding does not match model/site/readout/tokenizer/calibration")
    if artifact.get("config") != config:
        raise ValueError("artifact dimensions or algorithm configuration do not match")
    return artifact["state"]
