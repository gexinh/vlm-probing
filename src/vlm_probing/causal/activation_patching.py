"""Exact activation replacement; model execution is supplied by the caller."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from .base import BaseCausal, Runner, selection_mask, validate_pair


class ActivationPatching(BaseCausal):
    """Copy selected source entries into a same-shaped receiver activation."""

    name = "activation_patching"

    def run(
        self,
        receiver: Tensor,
        source: Tensor,
        *,
        mask: Tensor | None = None,
        runner: Runner | None = None,
    ) -> ProbeResult:
        validate_pair(receiver, source)
        selected = selection_mask(receiver, mask)
        edited = torch.where(selected, source, receiver)
        return self._intervention_result(
            receiver, edited, selected, runner=runner,
            metadata={"direction": "source_into_receiver"},
        )
