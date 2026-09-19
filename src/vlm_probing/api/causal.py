"""Independent layer interventions, scoring, and explicit edge attribution."""
import torch

from .. import causal as kernels
from ..adapters import CapabilityError
from ..core import ProbeResult
from .common import (BoundMethod, check_alignment, configuration, evaluate, layers_for,
                     metric_metadata, pack)


class CausalProbe(BoundMethod):
    def __init__(self, probe, name, layers, tokens="all", **options):
        super().__init__(probe, name, layers)
        self.tokens, self.options = tokens, options

    def run(self, inputs, *, metric, source=None, reference=None, direction=None,
            alignment="strict"):
        p, name, o = self.probe, self.name, self.options
        mapping = p.spec.attention_scores if name == "knockout" else p.spec.residuals
        sites = [mapping[i] for i in self.layers]
        paired = name in {"patch", "attribute"}
        if paired and source is None:
            raise ValueError(f"{name} requires source=clean_inputs")
        if name == "steer" and direction is None:
            raise ValueError("steer requires direction=... in residual coordinates")
        if name == "ablate" and o["mode"] != "zero" and reference is None:
            raise ValueError("mean/resample ablation requires reference=model_inputs")
        trace = p._run(inputs, capture=sites)
        layout = p._layout(inputs, trace.logits)
        baseline = evaluate(metric, trace.logits, layout)
        source_trace = None
        if paired or reference is not None:
            other = source if paired else reference
            source_trace = p._run(other, capture=sites)
            other_layout = p._layout(other, source_trace.logits)
            if paired or (name == "ablate" and o["mode"] == "resample"):
                check_alignment(p, other, inputs, other_layout, layout, alignment)
            elif name == "ablate" and not other_layout.valid.all():
                raise ValueError("mean reference must be unpadded; use the tensor kernel for custom masked means")
        token_mask = layout.select(self.tokens) if name != "knockout" else None
        results = []
        for layer, site in zip(self.layers, sites):
            original = trace.activations[site]
            if name != "knockout" and original.shape[:2] != layout.valid.shape:
                raise ValueError("causal residual must match the expanded token layout")
            mask = token_mask.to(original.device)[..., None] if token_mask is not None else None
            src = source_trace.activations[site] if source_trace is not None else None
            if name == "attribute":
                def metric_hidden(value):
                    output = p._run(inputs, interventions={site: lambda _: value}, grad=True)
                    return evaluate(metric, output.logits, layout).sum()
                result = kernels.AttributionPatching().run(original, src, mask=mask, metric=metric_hidden)
                # Retain useful scores, not full cached activations/gradients.
                results.append(ProbeResult(result.method, {
                    "token_scores": result.tensors["attribution"].sum(-1),
                    "estimated_effect": result.tensors["estimated_effect"],
                }, result.metadata))
                continue
            if name == "patch":
                edited = kernels.ActivationPatching().run(original, src, mask=mask).tensors["edited"]
            elif name == "ablate":
                edited = kernels.Ablation().run(original, mode=o["mode"], mask=mask,
                                                reference=src, mean_dims=o["mean_dims"]).tensors["edited"]
            elif name == "steer":
                value = direction[layer] if isinstance(direction, dict) else direction
                edited = kernels.Steering().run(original, value, mask=mask,
                                                strength=o["strength"], preserve_norm=o["preserve_norm"]).tensors["edited"]
            else:
                query = layout.select(o["queries"]).to(original.device)
                key = layout.select(o["keys"]).to(original.device)
                blocked = query[:, None, :, None] & key[:, None, None, :]
                valid_query = layout.valid.to(original.device)[:, None, :, None]
                safe = original.masked_fill(~valid_query, 0)
                edited = kernels.AttentionKnockout().run(safe, blocked=blocked).tensors["edited"]
                edited = torch.where(valid_query, edited, original)
            output = p._run(inputs, interventions={site: lambda _: edited})
            score = evaluate(metric, output.logits, layout)
            method_name = {"patch": "activation_patching", "ablate": "ablation",
                           "steer": "steering", "knockout": "attention_knockout"}[name]
            results.append(ProbeResult(method_name, {"baseline_score": baseline, "intervention_score": score,
                                              "effect": score - baseline}, {"effect_measured": True}))
        return pack(p, results[0].method, self.layers, results,
                    sweep="independent per-layer runs", alignment=alignment if paired else None,
                    configuration=configuration({"tokens": self.tokens, **o}),
                    metric=metric_metadata(metric),
                    effect_sign="source/intervention minus receiver")


class EdgeProbe(BoundMethod):
    def __init__(self, probe, edges, steps):
        super().__init__(probe, "eap_ig", [])
        self.edges, self.steps = edges, steps

    def run(self, inputs, *, source, metric, alignment="strict"):
        p = self.probe
        sites = [p.spec.edges[name] for name in self.edges]
        receiver = p._run(inputs, capture=sites)
        donor = p._run(source, capture=sites)
        layout = p._layout(inputs, receiver.logits)
        other = p._layout(source, donor.logits)
        check_alignment(p, source, inputs, other, layout, alignment)
        original = torch.stack([receiver.activations[site] for site in sites])
        clean = torch.stack([donor.activations[site] for site in sites])

        def replay(messages):
            edits = {site: (lambda _, i=i: messages[i]) for i, site in enumerate(sites)}
            output = p._run(inputs, interventions=edits, grad=True)
            return evaluate(metric, output.logits, layout).sum()

        result = kernels.EAPIG().run(original, clean, edge_names=self.edges, metric=replay, steps=self.steps)
        # Avoid returning full message caches through the convenient interface.
        result.tensors = {k: v for k, v in result.tensors.items() if k in {
            "edge_scores", "effect", "baseline_score", "intervention_score", "completeness_error"}}
        result.metadata["alignment"] = alignment
        result.metadata["metric"] = metric_metadata(metric)
        return self._finish(result)


class CausalMethods:
    def __init__(self, probe):
        self.probe = probe

    def patch(self, *, layers=None, tokens="visual"):
        return CausalProbe(self.probe, "patch", layers_for(self.probe, layers), tokens)

    def ablate(self, *, layers=None, tokens="visual", mode="zero", mean_dims=(0,)):
        if mode not in {"zero", "mean", "resample"}:
            raise ValueError("mode must be zero, mean, or resample")
        return CausalProbe(self.probe, "ablate", layers_for(self.probe, layers), tokens,
                           mode=mode, mean_dims=mean_dims)

    def attribute(self, *, layers=None, tokens="visual"):
        return CausalProbe(self.probe, "attribute", layers_for(self.probe, layers), tokens)

    def steer(self, *, layers=None, tokens="visual", strength=1., preserve_norm=False):
        return CausalProbe(self.probe, "steer", layers_for(self.probe, layers), tokens,
                           strength=strength, preserve_norm=preserve_norm)

    def knockout(self, *, layers=None, queries="text", keys="visual"):
        return CausalProbe(self.probe, "knockout", layers_for(self.probe, layers, "attention_scores"),
                           queries=queries, keys=keys)

    def eap_ig(self, *, edges=None, steps=32):
        edges = list(self.probe.spec.edges) if edges is None else list(edges)
        if not edges or any(name not in self.probe.spec.edges for name in edges):
            raise CapabilityError("EAP-IG requires explicit computational edge sites in spec.edges")
        if len(set(edges)) != len(edges):
            raise ValueError("edge names must be unique")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        return EdgeProbe(self.probe, edges, steps)
