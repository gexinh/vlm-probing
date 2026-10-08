"""Chen et al. attention perception and integrated reasoning feedback."""

import torch
from torch import Tensor

from ..core.types import ProbeResult
from ._attribution_utils import (finite_result, input_feedback, integration_metadata,
                                 matching_gradient, start_index)
from .base import BaseAttention


class BeyondIntuition(BaseAttention):
    """Beyond Intuition, head-wise or token-wise approximation.

    The official repository cites 2022; the published TMLR PDF is dated 2023.

    Head-wise: I_h = mean(abs(A_h.T @ dF/dA_h)); normalize I over heads;
    C = sum_h(I_h*A_h). Token-wise: C = mean_h(A_h) @ diag(||v_j W_O||/||z_j||),
    where z is the pre-LN residual input and v_j contains concatenated heads'
    value vectors; output projection bias is excluded. In either variant,
    start R=I and update R <- R+C@R in forward order. Finally multiply R by
    positive averaged final-attention gradients along the MODEL INPUT path.

    These are the official generator's Ours-H/Ours-C approximations, including
    value/output projections for the token variant. They are not gradient x
    attention, full Jacobians, or a layerwise relevance propagation rule.
    """

    name = "beyond_intuition"

    def __init__(self, *, variant: str = "head", start_layer: int = 0):
        if variant not in {"head", "token"}:
            raise ValueError("variant must be 'head' or 'token'")
        self.variant = variant
        self.start_layer = start_layer

    def run(self, attention: Tensor, integrated_gradients: Tensor, *,
            gradients: Tensor | None = None, input_norms: Tensor | None = None,
            projected_value_norms: Tensor | None = None,
            valid_tokens: Tensor | None = None, steps: int | None = None,
            quadrature: str = "provided") -> ProbeResult:
        """Return perception and relevance [batch,query,key] for all queries.

        Attention/gradients: [layer,batch,head,token,token]. Final integrated
        gradient: [batch,head,token,token]. Token norm inputs: [layer,batch,token].
        start_layer indexes this stack, zero-based inclusive. Each example's
        head normalization is independent; padding never alters its weights.
        """
        mask = self._stack(attention, valid_tokens)
        start_index(self.start_layer, attention.shape[0])
        integration = integration_metadata(steps, quadrature)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        if self.variant == "head":
            if gradients is None:
                raise ValueError("head variant requires original-forward attention gradients")
            matching_gradient(attention, gradients, "gradients")
        else:
            expected = attention.shape[:2] + attention.shape[-1:]
            for name, value in (("input_norms", input_norms),
                                ("projected_value_norms", projected_value_norms)):
                self._floating(value, name, 3)
                if (value.shape != expected or value.device != attention.device
                        or value.dtype != attention.dtype):
                    raise ValueError(f"{name} must match [layer,batch,token], device, and dtype")
                if not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError(f"{name} must be finite and nonnegative")
            if (input_norms[self.start_layer:].masked_select(mask[None]) <= 0).any():
                raise ValueError("valid pre-LN input norms must be positive")
        perception = torch.eye(attention.shape[-1], device=attention.device,
                               dtype=attention.dtype)[None] * mask[:, :, None]
        layer_perception, layer_weights = [], []
        for index in range(self.start_layer, attention.shape[0]):
            layer = attention[index].masked_fill(~pair_mask[:, None], 0)
            if self.variant == "head":
                gradient = gradients[index].masked_fill(~pair_mask[:, None], 0)
                importance = (layer.transpose(-2, -1) @ gradient).abs().sum((-2, -1))
                importance = importance / mask.sum(-1).square()[:, None]
                denominator = importance.sum(-1, keepdim=True)
                weights = importance / denominator.masked_fill(denominator == 0, 1)
                fused = (layer * weights[..., None, None]).sum(1)
            else:
                weights = projected_value_norms[index] / input_norms[index].masked_fill(~mask, 1)
                weights = weights.masked_fill(~mask, 0)
                fused = layer.mean(1) * weights[:, None, :]
            perception = perception + fused @ perception
            layer_perception.append(perception)
            layer_weights.append(weights)
        feedback = input_feedback(attention, integrated_gradients, mask)
        relevance = perception.abs() * feedback
        finite_result(relevance)
        tensors = {"relevance": relevance, "perception": perception,
                   "feedback": feedback, "layer_perception": torch.stack(layer_perception),
                   "head_weights" if self.variant == "head" else "token_weights": torch.stack(layer_weights)}
        return ProbeResult(self.name, tensors,
                           {**integration, "variant": self.variant, "start_layer": self.start_layer,
                            "integration_space": "model_input", "layer_order": "earliest_first",
                            "identity_included": True, "row_normalized": False,
                            "head_normalization": "independent_per_example" if self.variant == "head" else None,
                            "zero_head_gradient_rule": "zero_weight" if self.variant == "head" else None,
                            "token_norm_denominator": "block_pre_layernorm_residual_input" if self.variant == "token" else None,
                            "projected_values_include_output_bias": False if self.variant == "token" else None,
                            "reference": "https://openreview.net/forum?id=rm0zIzlhcX",
                            "official_code": "https://github.com/jiaminchen-1031/transformerinterp"})
