"""Persistent multi-batch training for existing Tuned/Attention Lens kernels."""
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ..lenses import AttentionLens, TunedLens


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def _atomic_torch(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def _parts(lens):
    if isinstance(lens, AttentionLens):
        return lens.decoders, lambda x: lens.decoders(x).sum(-2)
    if isinstance(lens, TunedLens):
        return lens.translator, lens._predict
    raise TypeError("distribution training supports TunedLens or AttentionLens kernels")


def _batch(batch, module):
    reference = next(module.parameters())
    inputs = batch["activations"].detach().to(reference)
    teacher = batch["teacher_logits"].detach().to(reference.device, torch.float32)
    mask = batch.get("mask")
    if mask is None:
        mask = torch.ones(teacher.shape[:-1], dtype=torch.bool, device=reference.device)
    else:
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(teacher.shape[:-1]):
            raise ValueError("mask must be boolean and align with teacher positions")
        mask = mask.to(reference.device)
    if not mask.any() or not torch.isfinite(inputs).all() or not torch.isfinite(teacher).all():
        raise ValueError("calibration batch must contain finite inputs and selected positions")
    return inputs, teacher, mask


def _losses(prediction, teacher):
    if prediction.shape != teacher.shape:
        raise ValueError("lens and teacher logits must have identical shapes")
    return F.kl_div(prediction.float().log_softmax(-1), teacher.log_softmax(-1),
                    reduction="none", log_target=True).sum(-1)


@torch.no_grad()
def evaluate_distribution_lens(lens, batches):
    """Return held-out teacher KL and full-vocabulary top-1 agreement."""
    module, predict = _parts(lens)
    total, agreement, count = 0.0, 0, 0
    for batch in batches:
        inputs, teacher, mask = _batch(batch, module)
        logits = predict(inputs)
        total += float(_losses(logits, teacher)[mask].sum())
        agreement += int((logits.argmax(-1)[mask] == teacher.argmax(-1)[mask]).sum())
        count += int(mask.sum())
    if not count:
        raise ValueError("validation iterable is empty")
    return {"kl": total / count, "top1_agreement": agreement / count, "positions": count}


def train_distribution_lens(lens, train_batches, validation_batches, checkpoint_dir, *,
                            epochs=10, lr=1e-3, patience=3, min_delta=1e-4,
                            seed=0, resume=True, provenance=None, progress_every=16):
    """Train a kernel with persistent Adam and held-out early stopping.

    Re-iterable DataLoaders yield dictionaries containing ``activations`` and
    ``teacher_logits`` plus an optional position ``mask``. Supply only the
    desired positions (e.g. final context positions for original LensA).
    KL is exactly ``KL(teacher || lens)``; Attention Lens sums head logits.
    Model/readout parameters never receive updates or accumulated gradients.

    ``best.pt`` uses the kernel's normal artifact format. ``training_state.pt``
    retains last-epoch kernel/Adam/RNG/loader states for resuming. Return history
    and restore the best kernel. Caller-provided provenance must match on resume.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError("calibration cannot run in torch.inference_mode()")
    if epochs < 1 or patience < 1 or lr <= 0 or min_delta < 0:
        raise ValueError("epochs/patience/lr must be positive and min_delta nonnegative")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    module, predict = _parts(lens)
    parameters = list(module.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    torch.manual_seed(seed)
    provenance = dict(provenance or {})
    configuration = {"lr": lr, "patience": patience, "min_delta": min_delta, "seed": seed}
    path = directory / "training_state.pt"
    generator = getattr(train_batches, "generator", None)
    start_epoch, stale, best = 0, 0, float("inf")
    history = {"method": lens.name, "binding": dict(lens.binding), "seed": seed,
        "objective": "KL(teacher || sum_head_logits)" if isinstance(lens, AttentionLens)
                     else "KL(teacher || lens)", "lr": lr, "epochs": [],
        "provenance": provenance, "configuration": configuration,
        "checkpoint": str(directory / "best.pt")}
    if resume and path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state["method"] != lens.name or state["binding"] != dict(lens.binding)
                or state["provenance"] != provenance or state["configuration"] != configuration):
            raise ValueError("resume checkpoint method, binding, or provenance does not match")
        module.load_state_dict(state["module"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
        if generator is not None and state.get("loader_rng") is not None:
            generator.set_state(state["loader_rng"])
        start_epoch, stale, best, history = state["epoch"], state["stale"], state["best"], state["history"]
        del state
    else:
        history["initial_validation"] = evaluate_distribution_lens(lens, validation_batches)
        best = history["initial_validation"]["kl"]
        history["best_epoch"] = 0
        temp = directory / "best.pt.tmp"
        lens.save(temp)
        os.replace(temp, directory / "best.pt")
    started = time.monotonic()
    for epoch in range(start_epoch, start_epoch if stale >= patience else epochs):
        total, count, steps = 0.0, 0, 0
        for step, batch in enumerate(train_batches):
            inputs, teacher, mask = _batch(batch, module)
            with torch.enable_grad():
                loss_by_position = _losses(predict(inputs), teacher)
                loss = loss_by_position[mask].mean()
                if not torch.isfinite(loss):
                    raise ValueError("calibration loss is not finite")
                gradients = torch.autograd.grad(loss, parameters)
                optimizer.zero_grad(set_to_none=True)
                for parameter, gradient in zip(parameters, gradients):
                    if not torch.isfinite(gradient).all():
                        raise ValueError("calibration gradient is not finite")
                    parameter.grad = gradient
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
            steps += 1
            if (step + 1) % progress_every == 0:
                _atomic_json(directory / "status.json", {"status": "training", "epoch": epoch + 1,
                    "steps_in_epoch": steps, "positions_in_epoch": count,
                    "train_kl": total / count, "elapsed_seconds": time.monotonic() - started})
        if not count:
            raise ValueError("training iterable is empty")
        validation = evaluate_distribution_lens(lens, validation_batches)
        row = {"epoch": epoch + 1, "train_kl": total / count, "train_positions": count,
               "optimizer_steps": steps, "validation": validation,
               "elapsed_seconds": time.monotonic() - started}
        history["epochs"].append(row)
        lens.is_fitted = True
        if validation["kl"] < best - min_delta:
            best, stale = validation["kl"], 0
            history["best_epoch"] = epoch + 1
            temp = directory / "best.pt.tmp"
            lens.save(temp)
            os.replace(temp, directory / "best.pt")
        else:
            stale += 1
        _atomic_json(directory / "history.json", history)
        _atomic_torch(path, {"format_version": 1, "method": lens.name,
            "binding": dict(lens.binding), "provenance": provenance, "configuration": configuration,
            "module": module.state_dict(), "optimizer": optimizer.state_dict(),
            "rng": torch.get_rng_state(), "loader_rng": generator.get_state() if generator is not None else None,
            "epoch": epoch + 1, "best": best, "stale": stale, "history": history})
        print(json.dumps({"method": lens.name, **row, "best_validation_kl": best}), flush=True)
        if stale >= patience:
            break
    if not (directory / "best.pt").exists():
        raise RuntimeError("no fitted checkpoint was produced")
    lens.load(directory / "best.pt")
    history["best_validation_kl"] = best
    history["status"] = "complete"
    history["completed_epochs"] = len(history["epochs"])
    _atomic_json(directory / "history.json", history)
    _atomic_json(directory / "status.json", {"status": "complete", "epochs": len(history["epochs"]),
        "best_validation_kl": best, "artifact": str(directory / "best.pt")})
    return history
