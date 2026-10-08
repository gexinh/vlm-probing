"""Validate cached image-conditioned activations before lens calibration.

The cache contains config/status JSON files and train/validation tensor shards.
A separate samples.json manifest binds their image IDs and split membership.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def validate_cache(cache_root, data_root, layers):
    """Reject partial, stale, or overlapping image splits before fitting a lens."""
    cache_root, data_root = Path(cache_root), Path(data_root)
    config = json.loads((cache_root / "config.json").read_text())
    status = json.loads((cache_root / "status.json").read_text())
    manifest_path = data_root / "samples.json"
    manifest = json.loads(manifest_path.read_text())
    if status.get("state") != "complete":
        raise ValueError("activation extraction must finish before training")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError("cached activations belong to a different dataset manifest")
    if not set(layers).issubset(config["layers"]):
        raise ValueError("requested layer is absent from the activation cache")
    all_ids, counts = set(), {}
    for split in ("train", "validation"):
        expected = [row["image_id"] for row in manifest["samples"] if row["split"] == split]
        actual, tokens = [], 0
        for path in sorted(cache_root.glob(f"{split}_*.pt")):
            chunk = torch.load(path, map_location="cpu", weights_only=True)
            actual.extend(chunk["image_ids"])
            count = len(chunk["teacher_logits"])
            if count != len(chunk["positions"]) or any(len(chunk["activations"][layer]) != count for layer in layers):
                raise ValueError("cached activation and teacher positions do not align")
            if not {item[0] for item in chunk["positions"]}.issubset(chunk["image_ids"]):
                raise ValueError("cached positions contain an unknown image")
            tokens += count
        if not expected or actual != expected or len(set(actual)) != len(actual):
            raise ValueError(f"cached {split} image IDs do not match the complete manifest")
        if all_ids.intersection(actual):
            raise ValueError("training and validation images overlap")
        if status["complete_images"].get(split) != len(expected):
            raise ValueError("cache status disagrees with the dataset manifest")
        all_ids.update(actual)
        counts[split] = {"images": len(actual), "prediction_positions": tokens}
    return config, counts
