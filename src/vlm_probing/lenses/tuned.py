"""Affine KL-calibrated lens: https://github.com/AlignmentResearch/tuned-lens."""
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from .base import (BaseLens, decoded, fit_distribution, floating_tensor,
                   load_artifact, save_artifact)
from ..core.types import ProbeResult


class TunedLens(BaseLens):
    """One site-specific affine translator followed by a frozen model readout.

    The residual parameterization ``h + translator(h)`` begins as identity.
    Construct one instance per layer/site. Calibration tensors and teachers
    must refer to matching token positions (no implicit next-token shift).
    ``dtype`` controls translator precision; ``readout_dtype`` independently
    controls model readout precision, e.g. fp32 translators with a bf16 head.
    Set device to the readout device when using a model on GPU.
    """

    name = "tuned_lens"

    def __init__(
        self, readout: Callable[[Tensor], Tensor], hidden_size: int, *,
        binding: Mapping[str, str] | None = None, device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32, readout_dtype: torch.dtype | None = None,
    ) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.readout, self.binding = readout, dict(binding or {})
        self.readout_dtype = readout_dtype or dtype
        self.translator = nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        nn.init.zeros_(self.translator.weight)
        nn.init.zeros_(self.translator.bias)
        self.is_fitted = False

    def _predict(self, activations: Tensor) -> Tensor:
        hidden = activations + self.translator(activations)
        return decoded(self.readout, hidden.to(dtype=self.readout_dtype))

    def fit(
        self, activations: Tensor, teacher_logits: Tensor, *, steps: int = 100,
        lr: float = 1e-3, mask: Tensor | None = None,
    ) -> list[float]:
        """Fit translator only; return pre-update KL values for each step."""
        losses = fit_distribution(self.translator, self._predict, activations,
                                  teacher_logits, steps=steps, lr=lr, mask=mask)
        self.is_fitted = True
        return losses

    @torch.no_grad()
    def run(self, activations: Tensor, *, layer: int | None = None) -> ProbeResult:
        floating_tensor(activations, "activations")
        if activations.shape[-1] != self.translator.in_features:
            raise ValueError("activation hidden width does not match this translator")
        activations = activations.to(self.translator.weight)
        return ProbeResult(self.name, {"logits": self._predict(activations)}, {
            "layer": layer, "fitted": self.is_fitted, "binding": dict(self.binding),
            "objective": "KL(teacher || lens)", "identity_initialized": True,
        })

    def save(self, path: str | Path) -> None:
        save_artifact(path, self.name, self.binding, self._config(),
                      {"translator": self.translator.state_dict(), "fitted": self.is_fitted})

    def load(self, path: str | Path) -> "TunedLens":
        state = load_artifact(path, self.name, self.binding, self._config())
        self.translator.load_state_dict(state["translator"])
        self.is_fitted = bool(state["fitted"])
        return self

    def _config(self) -> dict[str, object]:
        return {"hidden_size": self.translator.in_features, "readout_dtype": str(self.readout_dtype)}
