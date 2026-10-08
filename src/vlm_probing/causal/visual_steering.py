"""Per-image contrastive directions from VISTA's VSV construction (Eq. 4).

The returned vectors are consumed by the ordinary residual-steering API. This
module does not train the model or implement VISTA's separate SLA decoder.
"""
from collections.abc import Mapping

import torch

from ..adapters import CapabilityError


class VisualSteering:
    """A reusable, model-bound collection of per-layer visual directions.

    Positive and negative inputs must contain the same text context; the caller
    removes visual content when preparing the negative input. Sequence lengths
    may differ. Each input must select exactly one context-end position per
    example. For VSV, prepare a single image with a single question, without any
    answer tokens, and select the final prompt position.
    """

    def __init__(self, probe, directions, *, metadata=None):
        if not isinstance(directions, Mapping) or not directions:
            raise ValueError("directions must be a nonempty layer-to-tensor mapping")
        if any(type(layer) is not int or layer not in probe.spec.residuals
               for layer in directions):
            raise CapabilityError("directions must identify available residual layers")
        checked = {}
        batch = None
        for layer, vector in directions.items():
            if (not isinstance(vector, torch.Tensor) or vector.ndim != 2
                    or not vector.is_floating_point() or not torch.isfinite(vector).all()):
                raise ValueError("each direction must be a finite [batch, residual_dim] tensor")
            if not vector.shape[0] or not vector.shape[1]:
                raise ValueError("directions must have nonempty batch and residual axes")
            batch = vector.shape[0] if batch is None else batch
            if vector.shape[0] != batch:
                raise ValueError("directions must share a batch size")
            # Keep an explicit token axis for broadcasting over arbitrary prefixes.
            checked[layer] = vector.detach().float().cpu().clone()[:, None, :]
        self.probe, self.directions = probe, checked
        self.metadata = {"construction": "positive minus negative last-context residual",
                         "trained": False, "paper_equation": 4, "layers": sorted(checked),
                         **dict(metadata or {})}

    @classmethod
    def from_inputs(cls, probe, positive, negative, *, layers=None, tokens="last_prompt"):
        """Extract raw VSV directions in two real, unmodified model forwards.

        The last residual is captured before final normalization, consistently
        with the adapter's residual coordinate system and the steering sites.
        Equal sequence lengths are not required because visual tokens are absent
        from the negative context.
        """
        from ..api.common import layers_for
        chosen = layers_for(probe, layers)
        sites = [probe.spec.residuals[layer] for layer in chosen]
        traces = [probe._run(inputs, capture=sites) for inputs in (positive, negative)]
        layouts = [probe._layout(inputs, trace.logits)
                   for inputs, trace in zip((positive, negative), traces)]
        masks = [layout.select(tokens) for layout in layouts]
        if any(not torch.all(mask.sum(-1) == 1) for mask in masks):
            raise ValueError("VSV construction requires one selected context position per example")
        if masks[0].shape[0] != masks[1].shape[0]:
            raise ValueError("positive and negative contexts must have equal batch sizes")
        directions = {}
        for layer, site in zip(chosen, sites):
            vectors = []
            for trace, layout, mask in zip(traces, layouts, masks):
                hidden = trace.activations[site]
                if hidden.shape[:2] != layout.valid.shape:
                    raise ValueError("residual sites must align with expanded token positions")
                vectors.append(hidden[mask.to(hidden.device)].float())
            if vectors[0].shape != vectors[1].shape:
                raise ValueError("positive and negative residual coordinates must match")
            directions[layer] = vectors[0] - vectors[1]
        return cls(probe, directions, metadata={
            "actual_forward_passes": 2, "tokens": str(tokens),
            "positive_positions": masks[0].nonzero().cpu().tolist(),
            "negative_positions": masks[1].nonzero().cpu().tolist(),
            "model_id": probe.adapter.model_id,
        })

    def configure(self, *, strength=0.17, tokens="prediction"):
        """Configure simultaneous, norm-preserving residual interventions.

        All vectors are image-specific and remain fixed across generation.
        ``tokens='prediction'`` intervenes on the context's last token and all
        generated positions; full-prefix replay preserves earlier interventions
        while leaving other prompt/visual positions unchanged. Teacher-forced
        inputs must identify the original context with ``TokenLayout.prompt``.
        The supplied direction uses raw residual units;
        it is not replaced by the author's MLP-output/PCA implementation.
        Choose an explicit scale for the current model. The default is retained
        for compatibility and is not a calibrated or model-independent optimum.
        """
        return self.probe.causal.steer(layers=sorted(self.directions), tokens=tokens,
                                       strength=strength, preserve_norm=True, joint=True)
