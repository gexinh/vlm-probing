"""Attention observation and real value-path interventions."""
import torch

from .. import attention as kernels
from ..adapters import CapabilityError
from ..core import ProbeResult
from .common import BoundMethod, configuration, evaluate, layers_for, metric_metadata, pack, require


class AttentionProbe(BoundMethod):
    def __init__(self, probe, name, layers, **options):
        super().__init__(probe, name, layers)
        self.options = options

    def run(self, inputs, *, metric=None):
        p, name, o = self.probe, self.name, self.options
        mapping = p.spec.attention_scores if name == "temperature" else p.spec.attentions
        if name == "head_logits":
            mapping = p.spec.heads
        sites = [mapping[i] for i in self.layers]
        capture = list(sites)
        if name == "head_logits":
            final_site = p.spec.residuals[max(p.spec.residuals)]
            capture = list(dict.fromkeys([*capture, final_site]))
        if name in {"relevance", "reweight", "temperature"} and metric is None:
            raise ValueError(f"{name} requires a target metric")
        # Frozen model weights must still permit gradients through activations.
        edits = {}
        if name == "relevance" and p.spec.embeddings is not None:
            edits[p.spec.embeddings] = lambda x: x.detach().requires_grad_(True)
        trace = p._run(inputs, capture=capture, interventions=edits, grad=name == "relevance")
        layout = p._layout(inputs, trace.logits)
        matrices = [trace.activations[site] for site in sites]
        if name in {"rollout", "relevance"}:
            if self.layers != list(range(0, self.layers[-1] + 1)):
                raise ValueError("propagation requires consecutive layers from 0 in forward order")
            stack = torch.stack(matrices)
            valid = layout.valid.to(stack.device)
            if name == "rollout":
                result = kernels.AttentionRollout(**o).run(stack, valid_tokens=valid)
            else:
                score = evaluate(metric, trace.logits, layout).sum()
                grads = torch.autograd.grad(score, matrices)
                result = kernels.AttentionRelevance().run(stack, torch.stack(grads), valid_tokens=valid)
            result.metadata.update(configuration=configuration(o), metric=metric_metadata(metric))
            return self._finish(result)
        results = []
        if name == "profile":
            groups = o["groups"]
            if groups is None:
                groups = {"visual": "visual", "text": "text"} if layout.visual is not None else {}
            # Empty groups (text-only input) have a meaningful zero mass.
            group_masks = {}
            for key, selector in groups.items():
                if isinstance(selector, str) and selector in {"visual", "text"} and layout.visual is not None:
                    group_masks[key] = layout.visual if selector == "visual" else layout.valid & ~layout.visual
                else:
                    group_masks[key] = layout.select(selector)
            for matrix in matrices:
                result = kernels.AttentionProfile().run(
                    matrix, groups={k: v.to(matrix.device) for k, v in group_masks.items()},
                    query_mask=layout.select(o["queries"]).to(matrix.device),
                    key_mask=layout.valid.to(matrix.device))
                results.append(result)
        elif name == "head_logits":
            full = trace.activations[final_site]
            weight, scale, center = p.spec.linear_readout(full)
            for layer, raw in zip(self.layers, matrices):
                heads = p._heads(layer, raw).permute(0, 2, 1, 3)
                results.append(kernels.HeadLogitAttribution().run(
                    heads, weight.to(heads), fixed_scale=scale.to(heads), center=center))
        else:
            baseline = evaluate(metric, trace.logits, layout)
            query_mask = layout.select(o["queries"])
            key_mask = layout.select(o["keys"]) if name == "reweight" else None
            for layer, site in zip(self.layers, sites):
                def edit(value):
                    active = query_mask.to(value.device)[:, None, :, None]
                    if name == "temperature":
                        result = kernels.AttentionTemperature(o["temperature"]).run(
                            value, query_mask=layout.valid.to(value.device))
                        return torch.where(active, result.tensors["logits"], value)
                    weights = torch.where(key_mask.to(value.device)[:, None, None, :], o["weight"], 1.)
                    result = kernels.AttentionReweight(renormalize=o["renormalize"]).run(
                        value, weights, query_mask=query_mask.to(value.device),
                        key_mask=layout.valid.to(value.device))
                    return result.tensors["attention"]
                edited = p._run(inputs, interventions={site: edit})
                score = evaluate(metric, edited.logits, layout)
                results.append(ProbeResult(f"attention_{name}", {"baseline_score": baseline,
                                                 "intervention_score": score, "effect": score - baseline},
                                           {"effect_measured": True}))
        return pack(p, results[0].method, self.layers, results,
                    configuration=configuration(o), metric=metric_metadata(metric),
                    sweep="independent per-layer runs" if name in {"reweight", "temperature"} else "observation")


class AttentionMethods:
    def __init__(self, probe):
        self.probe = probe

    def profile(self, *, layers=None, queries="all", groups=None):
        return AttentionProbe(self.probe, "profile", layers_for(self.probe, layers, "attentions"),
                              queries=queries, groups=groups)

    def rollout(self, *, layers=None, head_reduction="mean", residual=True):
        return AttentionProbe(self.probe, "rollout", self._propagation_layers(layers),
                              head_reduction=head_reduction, residual=residual)

    def relevance(self, *, layers=None):
        return AttentionProbe(self.probe, "relevance", self._propagation_layers(layers))

    def _propagation_layers(self, layers):
        available = self.probe.describe()["methods"]["attention.rollout"]["sites"]
        if not available:
            raise CapabilityError("propagation requires attention layers starting at 0; "
                                  "hybrid linear-attention gaps cannot be skipped")
        chosen = layers_for(self.probe, available if layers is None else layers, "attentions")
        if not chosen or chosen != list(range(chosen[-1] + 1)):
            raise CapabilityError("propagation requires consecutive attention layers from 0; "
                                  "hybrid linear-attention gaps cannot be skipped")
        return chosen

    def head_logits(self, *, layers=None):
        require(self.probe.spec.linear_readout, "head attribution requires spec.linear_readout")
        return AttentionProbe(self.probe, "head_logits", layers_for(self.probe, layers, "heads"))

    def reweight(self, *, layers=None, queries="text", keys="visual", weight=2., renormalize=True):
        chosen = layers_for(self.probe, layers, "attentions")
        if not set(chosen) <= self.probe.spec.editable_attention:
            raise CapabilityError("reweight requires editable probabilities before A @ V; "
                                  "returned diagnostic attentions are not editable")
        if not isinstance(weight, (int, float)) or not 0 <= weight < float("inf"):
            raise ValueError("weight must be finite and nonnegative")
        return AttentionProbe(self.probe, "reweight", chosen, queries=queries, keys=keys,
                              weight=weight, renormalize=renormalize)

    def temperature(self, *, layers=None, queries="all", temperature=1.):
        kernels.AttentionTemperature(temperature)  # validate before model execution
        return AttentionProbe(self.probe, "temperature", layers_for(self.probe, layers, "attention_scores"),
                              queries=queries, temperature=temperature)
