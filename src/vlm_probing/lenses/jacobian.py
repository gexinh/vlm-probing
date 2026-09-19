"""Averaged causal transport: https://github.com/anthropics/jacobian-lens."""
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from torch import Tensor

from .base import BaseLens, decoded, floating_tensor, load_artifact, position_mask, save_artifact
from ..core.types import ProbeResult


class JacobianLens(BaseLens):
    """Fit a source-residual to final-residual average Jacobian at one site.

    ``downstream`` must be a differentiable causal continuation: [B,S,D] to
    [B,S,D_final], with no cross-example coupling. The estimator sums effects
    over valid current/future targets, averages valid sources within each
    prompt, then averages prompts equally. It is not a per-token gradient
    saliency score. This exact output-dimension loop is a small CPU/reference
    implementation; fitting a large VLM requires a separately budgeted job.
    """

    name = "jacobian_lens"
    estimator = "sum_valid_causal_targets_mean_sources_mean_prompts"

    def __init__(
        self, readout: Callable[[Tensor], Tensor], *, binding: Mapping[str, str] | None = None,
    ) -> None:
        self.readout, self.binding = readout, dict(binding or {})
        self.jacobian: Tensor | None = None
        self.calibration: dict[str, object] = {}

    def fit(
        self, activations: Tensor, downstream: Callable[[Tensor], Tensor], *,
        valid_mask: Tensor | None = None, source_mask: Tensor | None = None,
        target_mask: Tensor | None = None, skip_first: int = 16, exclude_last: bool = True,
        max_sources: int | None = None, seed: int = 0,
    ) -> "JacobianLens":
        """Estimate J[out,in]; explicit source/target masks restrict valid positions.

        The default excludes the first sixteen valid tokens and the final valid
        token of each prompt, matching the reference calibration convention.
        Use skip_first=0/exclude_last=False to retain those positions. Optional
        source subsampling draws only from the remaining valid source set.
        Call fit once on the supplied calibration batch; it replaces prior J.
        """
        if torch.is_inference_mode_enabled():
            raise ValueError("Jacobian calibration cannot run inside torch.inference_mode()")
        floating_tensor(activations, "activations", ndim=3)
        if not isinstance(skip_first, int) or skip_first < 0:
            raise ValueError("skip_first must be a nonnegative integer")
        if max_sources is not None and (not isinstance(max_sources, int) or max_sources < 1):
            raise ValueError("max_sources must be positive")
        shape = tuple(activations.shape[:2])
        valid = position_mask(valid_mask, shape, activations.device).clone()
        for batch in range(shape[0]):
            positions = valid[batch].nonzero(as_tuple=True)[0]
            valid[batch, positions[:skip_first]] = False
            if exclude_last and len(positions):
                valid[batch, positions[-1]] = False
        sources = valid & position_mask(source_mask, shape, activations.device)
        targets = valid & position_mask(target_mask, shape, activations.device)
        if not sources.any(-1).all() or not targets.any(-1).all():
            raise ValueError("every prompt must retain at least one source and target position")
        if max_sources is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            for batch in range(shape[0]):
                positions = sources[batch].nonzero(as_tuple=True)[0]
                if len(positions) > max_sources:
                    subset = torch.randperm(len(positions), generator=generator)[:max_sources]
                    sources[batch] = False
                    sources[batch, positions[subset.to(positions.device)]] = True
        with torch.enable_grad():
            hidden = activations.detach().clone().requires_grad_(True)
            final = downstream(hidden)
            floating_tensor(final, "downstream final residual", ndim=3)
            if tuple(final.shape[:2]) != shape or not final.requires_grad:
                raise ValueError("downstream must preserve batch/sequence and the differentiable graph")
            rows = []
            for dimension in range(final.shape[-1]):
                objective = (final[..., dimension] * targets).sum()
                gradient = torch.autograd.grad(
                    objective, hidden, retain_graph=dimension + 1 < final.shape[-1],
                )[0]
                # A causal continuation has zero influence on earlier targets.
                per_prompt = (gradient.float() * sources[..., None]).sum(1)
                per_prompt = per_prompt / sources.sum(1)[:, None]
                rows.append(per_prompt.mean(0))
        jacobian = torch.stack(rows).detach()
        floating_tensor(jacobian, "estimated Jacobian", ndim=2)
        self.jacobian = jacobian
        self.calibration = {
            "estimator": self.estimator, "n_prompts": shape[0],
            "source_counts": sources.sum(1).tolist(), "target_counts": targets.sum(1).tolist(),
            "skip_first": skip_first, "exclude_last": exclude_last,
            "max_sources": max_sources, "seed": seed,
            "source_positions": [row.nonzero(as_tuple=True)[0].tolist() for row in sources],
            "target_positions": [row.nonzero(as_tuple=True)[0].tolist() for row in targets],
        }
        return self

    @torch.no_grad()
    def run(self, activations: Tensor, *, layer: int | None = None) -> ProbeResult:
        floating_tensor(activations, "activations")
        if self.jacobian is None:
            raise RuntimeError("fit or load the Jacobian lens before running it")
        if activations.shape[-1] != self.jacobian.shape[-1]:
            raise ValueError("activations do not match the fitted source hidden width")
        # Preserve the caller's dtype for model readout, as in model inference.
        transported = activations @ self.jacobian.to(activations).T
        return ProbeResult(self.name, {"logits": decoded(self.readout, transported),
                                       "transported": transported}, {
            "layer": layer, "binding": dict(self.binding), **self.calibration,
        })

    def save(self, path: str | Path) -> None:
        if self.jacobian is None:
            raise RuntimeError("fit the Jacobian lens before saving it")
        save_artifact(path, self.name, self.binding, {"estimator": self.estimator},
                      {"jacobian": self.jacobian.cpu(), "calibration": self.calibration})

    def load(self, path: str | Path) -> "JacobianLens":
        state = load_artifact(path, self.name, self.binding, {"estimator": self.estimator})
        floating_tensor(state["jacobian"], "artifact Jacobian", ndim=2)
        self.jacobian = state["jacobian"]
        self.calibration = dict(state["calibration"])
        return self
