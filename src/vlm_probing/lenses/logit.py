"""Untrained logit lens (https://github.com/AlignmentResearch/tuned-lens)."""
from collections.abc import Callable

import torch
from torch import Tensor

from .base import BaseLens, decoded, floating_tensor
from ..core.types import ProbeResult


class LogitLens(BaseLens):
    """Decode ``[..., hidden]`` through the model's actual final norm + head.

    ``readout`` must include the final normalization exactly once. Use residual
    states before that normalization; do not normalize an already normalized
    final state again.
    """

    name = "logit_lens"

    def __init__(self, readout: Callable[[Tensor], Tensor]) -> None:
        self.readout = readout

    @torch.no_grad()
    def run(self, activations: Tensor, *, layer: int | None = None) -> ProbeResult:
        floating_tensor(activations, "activations")
        logits = decoded(self.readout, activations)
        return ProbeResult(self.name, {"logits": logits}, {"layer": layer, "trained": False})
