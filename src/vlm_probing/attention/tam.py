"""Transition Attention Maps with input-path gradient feedback."""

from torch import Tensor

from ..core.types import ProbeResult
from ._attribution_utils import finite_result, input_feedback, integration_metadata, start_index
from .base import BaseAttention


class TransitionAttentionMaps(BaseAttention):
    """Yuan et al.: propagate the final attention row backwards through layers.

    Initialize s = mean_heads(A_last)[query,:]. For earlier layers in reverse
    order, s <- (s + s @ mean_heads(A_l))/2. Then multiply by
    mean_heads(relu(integral_0^1 dF(alpha*X)/dA_last(alpha*X) d alpha)).

    Here X is the model input tensor (normalized pixels in the original ViT
    work). This is an INPUT path, not Hao's attention-probability IG path.
    All query rows are returned so CLS/last-token selection remains explicit.
    """

    name = "transition_attention_maps"

    def __init__(self, *, start_layer: int = 0, residual_normalize: bool = True):
        self.start_layer = start_layer
        self.residual_normalize = residual_normalize

    def run(self, attention: Tensor, integrated_gradients: Tensor, *,
            valid_tokens: Tensor | None = None, steps: int | None = None,
            quadrature: str = "provided") -> ProbeResult:
        """Attention is [layer,batch,head,token,token], earliest layer first.

        Integrated gradients are [batch,head,token,token] for the final layer.
        start_layer is zero-based and inclusive among earlier transitions.
        residual_normalize=False reproduces the official code's missing 1/2;
        this only changes a global factor, not a normalized saliency map.
        """
        mask = self._stack(attention, valid_tokens)
        start_index(self.start_layer, attention.shape[0])
        integration = integration_metadata(steps, quadrature)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        mean_attention = attention.mean(2).masked_fill(~pair_mask[None], 0)
        state = mean_attention[-1]
        for layer in range(attention.shape[0] - 2, self.start_layer - 1, -1):
            state = state + state @ mean_attention[layer]
            if self.residual_normalize:
                state = state * 0.5
        feedback = input_feedback(attention, integrated_gradients, mask)
        relevance = state * feedback
        finite_result(relevance)
        return ProbeResult(self.name, {"relevance": relevance, "state": state,
                                      "feedback": feedback},
                           {**integration, "integration_space": "model_input",
                            "start_layer": self.start_layer, "layer_order": "latest_first",
                            "residual_normalized": self.residual_normalize,
                            "feedback_relu": "after_integration_before_head_mean",
                            "reference": "https://openreview.net/forum?id=TT-cf6QSDaQ",
                            "official_code": "https://github.com/XianrenYty/Transition_Attention_Maps"})
