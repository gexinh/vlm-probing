"""Grad-CAM over attention heads, as used by the official ViT baselines."""

from torch import Tensor

from ..core.types import ProbeResult
from ._attribution_utils import finite_result, matching_gradient
from .base import BaseAttention


class AttentionGradCAM(BaseAttention):
    """Use heads as Grad-CAM channels for each selected attention query.

    For query q, alpha[h,q] = mean_selected_keys(dS/dA[h,q,k]) and
    CAM[q,k] = relu(mean_heads(alpha[h,q] * A[h,q,k])). The comparison's
    ViT baseline selects the final CLS query and excludes the CLS key.
    This is an attention-map Grad-CAM variant, not CAM over image features.
    """

    name = "attention_grad_cam"

    def run(self, attention: Tensor, gradients: Tensor, *,
            query_mask: Tensor | None = None, key_mask: Tensor | None = None,
            normalize: bool = False) -> ProbeResult:
        """Return CAM [batch,query,key]; key_mask selects averaging/display keys.

        Masks select the map; attention is not renormalized over selected keys.
        Pass gradients from this attention's forward and an explicit target.
        """
        qmask, _ = self._probabilities(attention, query_mask)
        batch, _, _, keys = attention.shape
        kmask = self._mask(key_mask, (batch, keys), attention, "key_mask")
        if not kmask.any(-1).all():
            raise ValueError("each example needs at least one selected key")
        matching_gradient(attention, gradients, "gradients")
        selected = kmask[:, None, None, :]
        head_weights = gradients.masked_fill(~selected, 0).sum(-1)
        head_weights = head_weights / kmask.sum(-1)[:, None, None]
        head_weights = head_weights.masked_fill(~qmask[:, None, :], 0)
        cam = (attention * head_weights[..., None]).mean(1).clamp_min(0)
        pair_mask = qmask[:, :, None] & kmask[:, None, :]
        cam = cam.masked_fill(~pair_mask, 0)
        if normalize:
            minimum = cam.masked_fill(~kmask[:, None, :], float("inf")).amin(-1, keepdim=True)
            maximum = cam.amax(-1, keepdim=True)
            width = maximum - minimum
            cam = (cam - minimum) / width.masked_fill(width == 0, 1)
            cam = cam.masked_fill(~pair_mask, 0)
        finite_result(cam)
        return ProbeResult(self.name, {"cam": cam, "head_weights": head_weights},
                           {"variant": "attention_heads_as_channels",
                            "gradient_pooling": "selected_keys_per_query",
                            "head_reduction": "mean", "relu_after_head_reduction": True,
                            "normalized_for_display": normalize,
                            "reference": "https://arxiv.org/abs/1610.02391",
                            "attention_baseline_code": "https://github.com/hila-chefer/Transformer-Explainability"})
