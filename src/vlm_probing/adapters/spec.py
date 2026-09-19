"""Explicit sequence semantics and model capabilities for the public API."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


class CapabilityError(ValueError):
    """The adapter cannot provide the computation requested by a method."""


@dataclass
class TokenLayout:
    """Masks use the model's expanded sequence coordinates, including padding.

    Without ``prompt``, all valid positions are treated as prompt positions.
    Set it explicitly when inputs include a teacher-forced answer. ``token_ids``
    is optional but required by sequence scoring and automatic pair alignment.
    """

    valid: Tensor
    visual: Tensor | None = None
    prompt: Tensor | None = None
    token_ids: Tensor | None = None

    def checked(self, shape, device) -> "TokenLayout":
        values = {}
        for name in ("valid", "visual", "prompt"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, Tensor) or value.dtype != torch.bool or tuple(value.shape) != tuple(shape):
                    raise ValueError(f"layout.{name} must be boolean {tuple(shape)}")
                values[name] = value.to(device)
        if "valid" not in values:
            raise ValueError("layout.valid is required")
        valid = values["valid"]
        if not valid.any(-1).all():
            raise ValueError("each sequence must have at least one valid token")
        for name in ("visual", "prompt"):
            if name in values and (values[name] & ~valid).any():
                raise ValueError(f"layout.{name} selects padding")
        ids = self.token_ids
        if ids is not None:
            if not isinstance(ids, Tensor) or ids.dtype != torch.long or tuple(ids.shape) != tuple(shape):
                raise ValueError("layout.token_ids must be aligned long token IDs")
            ids = ids.to(device)
        return TokenLayout(**values, token_ids=ids)

    def select(self, tokens="all") -> Tensor:
        valid = self.valid
        if isinstance(tokens, str):
            if tokens == "all":
                result = valid.clone()
            elif tokens in {"visual", "text"}:
                if self.visual is None:
                    raise CapabilityError("visual/text selection requires an explicit visual token layout")
                result = self.visual if tokens == "visual" else valid & ~self.visual
            elif tokens == "last_prompt":
                prompt = valid if self.prompt is None else self.prompt
                if not prompt.any(-1).all():
                    raise ValueError("each example needs a prompt token")
                index = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
                last = index.masked_fill(~prompt, -1).max(-1).values
                result = torch.zeros_like(valid).scatter_(1, last[:, None], True)
            else:
                raise ValueError(f"unknown token selector {tokens!r}")
        elif isinstance(tokens, Tensor) and tokens.dtype == torch.bool:
            if tokens.shape != valid.shape:
                raise ValueError("token mask must match the expanded sequence")
            result = tokens.to(valid.device)
        else:
            positions = [tokens] if isinstance(tokens, int) else list(tokens)
            if not positions or any(type(p) is not int or not 0 <= p < valid.shape[1] for p in positions):
                raise ValueError("token positions must be nonnegative expanded-sequence indices")
            result = torch.zeros_like(valid)
            result[:, positions] = True
        if (result & ~valid).any():
            raise ValueError("token selector includes padding")
        if not result.any():
            raise ValueError("token selection is empty")
        return result


@dataclass
class ProbeInputs:
    """Native model kwargs plus optional explicit expanded-token metadata."""

    kwargs: Mapping[str, Any]
    layout: TokenLayout | None = None


@dataclass
class ModelSpec:
    """Site names reference an adapter's hook table; no shape-based inference.

    Head sites must produce [B,T,H,D_residual], or ``project_heads`` must convert
    them to that space. Attention sites are [B,H,Q,K]. Only names explicitly in
    ``editable_attention`` may be used for probability interventions. ``edges``
    declares independently replaceable messages, not arbitrary node caches.
    """

    residuals: dict[int, str]
    attentions: dict[int, str] = field(default_factory=dict)
    attention_scores: dict[int, str] = field(default_factory=dict)
    heads: dict[int, str] = field(default_factory=dict)
    embeddings: str | None = None
    input_embeddings: Tensor | None = None
    layout: Callable[[Mapping[str, Any]], TokenLayout] | None = None
    project_heads: Callable[[int, Tensor], Tensor] | None = None
    linear_readout: Callable[[Tensor], tuple[Tensor, Tensor, bool]] | None = None
    editable_attention: set[int] = field(default_factory=set)
    edges: dict[str, str] = field(default_factory=dict)
    # Metadata compared before automatic aligned patching (e.g. visual grids).
    alignment_keys: tuple[str, ...] = ()
    forward_defaults: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def validate(self, sites):
        if not self.residuals or any(type(k) is not int or k < 0 for k in self.residuals):
            raise ValueError("spec.residuals requires nonnegative layer indices")
        mappings = (self.residuals, self.attentions, self.attention_scores, self.heads, self.edges)
        names = {name for mapping in mappings for name in mapping.values()}
        if self.embeddings is not None:
            names.add(self.embeddings)
        missing = names - set(sites)
        if missing:
            raise ValueError(f"spec references unknown hook sites: {sorted(missing)}")
        if not self.editable_attention <= self.attentions.keys():
            raise ValueError("editable attention layers must declare probability sites")
        for mapping in (self.attentions, self.attention_scores, self.heads):
            if not mapping.keys() <= self.residuals.keys():
                raise ValueError("attention/head layer indices must exist in residuals")
