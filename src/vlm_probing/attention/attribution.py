"""Hao et al. ATTATTR: integrated gradients in attention-probability space."""

from torch import Tensor

from ..core.types import ProbeResult
from ._attribution_utils import finite_result, integration_metadata, matching_gradient
from .base import BaseAttention


class AttentionAttribution(BaseAttention):
    """ATTATTR = A * integral_0^1 dF(alpha*A)/d(alpha*A) d alpha.

    The input is fixed. All heads in one chosen layer are replaced together
    by alpha*A before multiplication by V. Other layers recompute normally.
    Scaled attention is deliberately NOT row-normalized; the zero baseline
    disconnects attention within this layer. Preserve the attribution sign.

    The tensor kernel consumes a pre-averaged integral, not endpoint gradients.
    The model-bound API performs actual probability-space interventions.
    Paper Eq.4 uses right Riemann sums; author code uses left Riemann sums.
    """

    name = "attention_attribution"

    def run(self, attention: Tensor, integrated_gradients: Tensor, *,
            query_mask: Tensor | None = None, key_mask: Tensor | None = None,
            steps: int | None = None, quadrature: str = "provided") -> ProbeResult:
        """Input is [batch,head,query,key]; masks select returned attribution."""
        qmask, _ = self._probabilities(attention, query_mask)
        kmask = self._mask(key_mask, (attention.shape[0], attention.shape[-1]),
                           attention, "key_mask")
        matching_gradient(attention, integrated_gradients, "integrated_gradients")
        integration = integration_metadata(steps, quadrature)
        attribution = attention * integrated_gradients
        pair_mask = qmask[:, None, :, None] & kmask[:, None, None, :]
        attribution = attribution.masked_fill(~pair_mask, 0)
        finite_result(attribution)
        return ProbeResult(self.name,
                           {"attribution": attribution,
                            "head_attribution": attribution.sum((-2, -1)),
                            "token_attribution": attribution.sum(1)},
                           {**integration, "integration_space": "one_layer_attention_probabilities",
                            "baseline": "zero_attention", "input_fixed": True,
                            "row_renormalized": False, "signed": True,
                            "reference": "https://arxiv.org/abs/2004.11207",
                            "official_code": "https://github.com/YRdddream/attattr"})
