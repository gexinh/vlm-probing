"""Exact grouped VJPs and weighted streaming Jacobian Lens calibration."""
import json
import time
from pathlib import Path

import torch

from .distribution import _atomic_json, _atomic_torch


def _masks(valid, skip_first, exclude_last):
    retained = valid.clone()
    for row in range(len(retained)):
        positions = valid[row].nonzero(as_tuple=True)[0]
        retained[row, positions[:skip_first]] = False
        if exclude_last and len(positions):
            retained[row, positions[-1]] = False
    if not retained.any(-1).all():
        raise ValueError("each calibration prompt must retain a valid source/target")
    return retained


def jacobian_sums_from_graph(final, sources, source_mask, target_mask, *, dim_batch=32):
    """Return sum of per-prompt J[out,in] for each graph source.

    Each row sums target cotangents, averages source positions within a prompt,
    then sums prompts. The caller divides by the total prompt count across
    batches, preserving equal-prompt weighting even for a short final batch.
    ``is_grads_batched`` groups exact basis-vector VJPs, not random projections.
    Multiple intermediate sources share one downstream graph.
    """
    if dim_batch < 1 or not final.requires_grad or not source_mask.any(-1).all():
        raise ValueError("positive dim_batch, differentiable graph, and nonempty sources required")
    names, tensors = list(sources), list(sources.values())
    width = final.shape[-1]
    output = {name: torch.empty(width, value.shape[-1], dtype=torch.float64)
              for name, value in sources.items()}
    for start in range(0, width, dim_batch):
        stop = min(start + dim_batch, width)
        cotangents = final.new_zeros((stop - start, *final.shape))
        for offset, dimension in enumerate(range(start, stop)):
            cotangents[offset, ..., dimension] = target_mask
        gradients = torch.autograd.grad(final, tensors, grad_outputs=cotangents,
            is_grads_batched=True, retain_graph=stop < width)
        for name, gradient in zip(names, gradients):
            per_prompt = (gradient.float() * source_mask[None, ..., None]).sum(2)
            per_prompt = per_prompt / source_mask.sum(-1)[None, :, None]
            output[name][start:stop] = per_prompt.double().sum(1).cpu()
    return output


def fit_jacobian_lenses(probe, input_batches, *, layers, binding, directory,
                        dim_batch=32, skip_first=16, exclude_last=True,
                        tokens="text", resume=True, checkpoint_every=16,
                        snapshot_counts=(64, 128, 256)):
    """Fit selected public Jacobian lenses from independent prompt batches.

    Uses native adapter sites, one graph per batch, exact grouped VJPs, and
    weighted merging. Model parameters and mode flags are restored afterwards.
    Inputs must be deterministic/re-iterable for resume; provide immutable data
    identity in binding. Artifacts use the normal public `.load(directory)`.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("Jacobian calibration cannot run under inference_mode")
    method = probe.lens.jacobian(layers=layers, tokens=tokens, binding=binding)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "calibration_state.pt"
    sums, prompts, completed_batches = {}, 0, 0
    settings = {"dim_batch": dim_batch, "skip_first": skip_first, "exclude_last": exclude_last,
                "tokens": tokens, "layers": method.layers, "binding": dict(binding),
                "snapshot_counts": list(snapshot_counts)}
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError("Jacobian resume settings/binding differ")
        sums, prompts, completed_batches = state["sums"], state["prompts"], state["batches"]
    earliest = probe.spec.residuals[min(method.layers)]
    final_site = probe.spec.residuals[max(probe.spec.residuals)]
    sites = list(dict.fromkeys([probe.spec.residuals[i] for i in method.layers] + [final_site]))
    flags = [parameter.requires_grad for parameter in probe.model.parameters()]
    modes = [(module, module.training) for module in probe.model.modules()]
    started = time.monotonic()
    checkpoints = []
    try:
        probe.model.eval()
        probe.model.requires_grad_(False)
        for batch_index, inputs in enumerate(input_batches):
            if batch_index < completed_batches:
                continue
            trace = probe._run(inputs, capture=sites, grad=True,
                interventions={earliest: lambda value: value.detach().requires_grad_(True)})
            layout = probe._layout(inputs, trace.logits)
            retained = _masks(layout.valid, skip_first, exclude_last)
            sources = retained & layout.select(tokens)
            final = trace.activations[final_site]
            values = {layer: trace.activations[probe.spec.residuals[layer]] for layer in method.layers}
            batch_sums = jacobian_sums_from_graph(final, values, sources, retained, dim_batch=dim_batch)
            for layer, value in batch_sums.items():
                sums[layer] = sums.get(layer, torch.zeros_like(value)) + value
            prompts += int(final.shape[0])
            completed_batches = batch_index + 1
            del trace, final, values, batch_sums
            if prompts in snapshot_counts:
                _atomic_torch(root / f"prefix_{prompts}.pt", {"prompts": prompts,
                    "jacobians": {layer: (value / prompts).float() for layer, value in sums.items()}})
            if completed_batches % checkpoint_every == 0:
                _atomic_torch(state_path, {"settings": settings, "sums": sums,
                    "prompts": prompts, "batches": completed_batches})
                checkpoints.append(prompts)
                _atomic_json(root / "status.json", {"status": "calibrating", "prompts": prompts,
                    "batches": completed_batches, "elapsed_seconds": time.monotonic() - started})
                print(json.dumps({"jacobian_prompts": prompts, "batches": completed_batches,
                    "elapsed_seconds": time.monotonic() - started}), flush=True)
    finally:
        for parameter, flag in zip(probe.model.parameters(), flags):
            parameter.requires_grad_(flag)
        for module, training in modes:
            module.training = training
    if not prompts:
        raise ValueError("Jacobian calibration iterable is empty")
    _atomic_torch(state_path, {"settings": settings, "sums": sums,
        "prompts": prompts, "batches": completed_batches})
    embedding = probe.spec.input_embeddings
    for layer in method.layers:
        kernel = method._new_kernel(layer, embedding.shape[-1], embedding.device, embedding.dtype)
        kernel.jacobian = (sums[layer] / prompts).float()
        kernel.calibration = {"estimator": kernel.estimator, "n_prompts": prompts,
            "n_batches": completed_batches, "dim_batch": dim_batch, "skip_first": skip_first,
            "exclude_last": exclude_last, "source_selector": tokens,
            "calibration_mode": "exact_grouped_vjp_weighted_batch_mean"}
        method.kernels[layer] = kernel
    method.save(root)
    metadata = {"status": "complete", "settings": settings, "prompts": prompts,
        "batches": completed_batches, "elapsed_seconds_this_run": time.monotonic() - started,
        "checkpoint_prompt_counts": checkpoints,
        "prefix_artifacts": [path.name for path in sorted(root.glob("prefix_*.pt"))]}
    _atomic_json(root / "status.json", metadata)
    return method, metadata
