"""Learned head decoders: https://github.com/msakarvadia/AttentionLens."""
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn

from .base import BaseLens, fit_distribution, floating_tensor, load_artifact, save_artifact
from ..core.types import ProbeResult


class _HeadDecoders(nn.Module):
    def __init__(self, heads: int, width: int, vocabulary: int, **kwargs: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, width, vocabulary, **kwargs))
        self.bias = nn.Parameter(torch.zeros(heads, vocabulary, **kwargs))
        nn.init.normal_(self.weight, std=width ** -0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.einsum("...hd,hdv->...hv", inputs, self.weight) + self.bias


class AttentionLens(BaseLens):
    """Per-head vocabulary maps fitted jointly through their summed logits.

    Input is ``[..., heads, head_size]``. The official LensA uses each head's
    post-output-projection contribution (head_size=d_model), initializes maps
    from unembedding, and calibrates the final position. ``fit`` defaults to
    final valid positions, while an explicit mask enables broader calibration.
    The optional pre_output_projection input space is an explicit extension.
    Individual head maps are learned decoders, not an additive causal score.
    """

    name = "attention_lens"

    def __init__(
        self, num_heads: int, head_size: int, vocab_size: int, *,
        binding: Mapping[str, str] | None = None,
        input_space: str = "post_output_projection",
        initial_unembedding: Tensor | None = None, initial_bias: Tensor | None = None,
        device: str | torch.device | None = None, dtype: torch.dtype = torch.float32,
    ) -> None:
        if min(num_heads, head_size, vocab_size) < 1:
            raise ValueError("head count, head width and vocabulary size must be positive")
        if input_space not in {"post_output_projection", "pre_output_projection"}:
            raise ValueError("input_space must specify pre_output_projection or post_output_projection")
        self.binding, self.input_space = dict(binding or {}), input_space
        self.decoders = _HeadDecoders(num_heads, head_size, vocab_size, device=device, dtype=dtype)
        self.config = {"num_heads": num_heads, "head_size": head_size,
                       "vocab_size": vocab_size, "input_space": input_space}
        self.is_fitted = False
        with torch.no_grad():
            if initial_unembedding is not None:
                floating_tensor(initial_unembedding, "initial_unembedding", ndim=2)
                if initial_unembedding.shape != (vocab_size, head_size):
                    raise ValueError("initial_unembedding must be [vocab_size, head_size]")
                self.decoders.weight.copy_(initial_unembedding.T)
            if initial_bias is not None:
                floating_tensor(initial_bias, "initial_bias", ndim=1)
                if initial_bias.shape != (vocab_size,):
                    raise ValueError("initial_bias must be [vocab_size]")
                self.decoders.bias.copy_(initial_bias)

    def fit(
        self, activations: Tensor, teacher_logits: Tensor, *, steps: int = 100,
        lr: float = 1e-3, mask: Tensor | None = None, valid_mask: Tensor | None = None,
    ) -> list[float]:
        """Fit sum-of-head logits to teacher logits with teacher-to-lens KL.

        For [B,S,H,D] input, absent an explicit mask, select the last valid
        token of each sequence; valid_mask handles right/left padding.
        """
        self._validate(activations)
        if mask is not None and valid_mask is not None:
            raise ValueError("provide mask or valid_mask, not both")
        if mask is None:
            if activations.ndim != 4:
                raise ValueError("default last-position calibration requires [batch, sequence, heads, width]")
            shape = tuple(activations.shape[:2])
            if valid_mask is None:
                valid_mask = torch.ones(shape, dtype=torch.bool, device=activations.device)
            if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != shape or not valid_mask.any(-1).all():
                raise ValueError("valid_mask must select at least one token per sequence")
            valid_mask = valid_mask.to(activations.device)
            positions = torch.arange(shape[1], device=activations.device).expand(shape)
            last = positions.masked_fill(~valid_mask, -1).max(-1).values
            mask = torch.zeros_like(valid_mask)
            mask.scatter_(1, last[:, None], True)
        losses = fit_distribution(self.decoders, lambda x: self.decoders(x).sum(-2),
                                  activations, teacher_logits, steps=steps, lr=lr, mask=mask)
        self.is_fitted = True
        return losses

    def _validate(self, activations: Tensor) -> None:
        floating_tensor(activations, "activations")
        if activations.ndim < 3 or tuple(activations.shape[-2:]) != (self.config["num_heads"], self.config["head_size"]):
            raise ValueError("activations must have shape [..., num_heads, head_size]")

    @torch.no_grad()
    def run(self, activations: Tensor, *, layer: int | None = None) -> ProbeResult:
        self._validate(activations)
        head_logits = self.decoders(activations.to(self.decoders.weight))
        return ProbeResult(self.name, {"head_logits": head_logits, "logits": head_logits.sum(-2)}, {
            "layer": layer, "fitted": self.is_fitted, "input_space": self.input_space,
            "binding": dict(self.binding), "objective": "KL(teacher || sum_head_logits)",
        })

    def save(self, path: str | Path) -> None:
        save_artifact(path, self.name, self.binding, self.config,
                      {"decoders": self.decoders.state_dict(), "fitted": self.is_fitted})

    def load(self, path: str | Path) -> "AttentionLens":
        state = load_artifact(path, self.name, self.binding, self.config)
        self.decoders.load_state_dict(state["decoders"])
        self.is_fitted = bool(state["fitted"])
        return self
