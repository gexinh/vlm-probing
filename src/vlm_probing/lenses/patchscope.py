"""Prompt-mediated readout: https://github.com/PAIR-code/interpretability."""
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

from .base import BaseLens, floating_tensor
from ..core.types import ProbeResult


class Patchscope(BaseLens):
    """Patch one source representation per example into an explanation prompt.

    ``runner(patch, *, target_layer, target_position, target_inputs)`` must
    install ``patch[B,D_target]`` at the requested target model site, run the
    supplied prompt, remove the patch hook, and return teacher-forced logits
    ``[B,T,V]``. The library's TorchModelAdapter can supply that callback.
    Model-specific generation is intentionally not assumed by this interface.

    ``mapping`` optionally aligns source and target representation spaces.
    Without it, the caller must use a compatible representation basis; equal
    tensor widths alone do not establish cross-model alignment.
    """

    name = "patchscope"

    def __init__(
        self, runner: Callable[..., Tensor], *, mapping: Callable[[Tensor], Tensor] | None = None,
        source_model_id: str | None = None, target_model_id: str | None = None,
    ) -> None:
        if mapping is None and source_model_id and target_model_id and source_model_id != target_model_id:
            raise ValueError("cross-model Patchscope requires an explicit alignment mapping")
        self.runner, self.mapping = runner, mapping
        self.source_model_id, self.target_model_id = source_model_id, target_model_id

    @torch.no_grad()
    def run(
        self, activations: Tensor, *, source_position: int, target_position: int,
        target_layer: int, target_inputs: Mapping[str, Any], layer: int | None = None,
    ) -> ProbeResult:
        floating_tensor(activations, "activations", ndim=3)
        if not isinstance(source_position, int) or not 0 <= source_position < activations.shape[1]:
            raise ValueError("source_position must index the supplied source sequence")
        if not isinstance(target_position, int) or target_position < 0:
            raise ValueError("target_position must be a nonnegative target sequence index")
        if not isinstance(target_layer, int) or target_layer < 0:
            raise ValueError("target_layer must be nonnegative")
        patch = activations[:, source_position].detach().clone()
        if self.mapping is not None:
            patch = self.mapping(patch)
        floating_tensor(patch, "mapped patch", ndim=2)
        if patch.shape[0] != activations.shape[0]:
            raise ValueError("mapping must preserve the batch dimension")
        logits = self.runner(patch, target_layer=target_layer, target_position=target_position,
                             target_inputs=dict(target_inputs))
        floating_tensor(logits, "target logits", ndim=3)
        if logits.shape[0] != activations.shape[0] or target_position >= logits.shape[1]:
            raise ValueError("runner logits must match source batch and contain the target position")
        return ProbeResult(self.name, {"logits": logits}, {
            "layer": layer, "source_position": source_position, "target_layer": target_layer,
            "target_position": target_position, "readout": "teacher_forced_target_prompt",
            "source_model_id": self.source_model_id, "target_model_id": self.target_model_id,
            "mapped": self.mapping is not None,
        })
