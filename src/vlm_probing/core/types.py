"""Small shared result types, independent of any model implementation."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import torch
from torch import Tensor


@dataclass
class ProbeResult:
    method: str
    tensors: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        """Save portable CPU tensors and JSON metadata without serializing models."""
        metadata = json.dumps({"method": self.method, **self.metadata}, indent=2,
                              ensure_ascii=False, allow_nan=False)
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.tensors.items()}, path / "tensors.pt")
        (path / "metadata.json").write_text(metadata + "\n", encoding="utf-8")


@dataclass
class ForwardTrace:
    output: Any
    activations: dict[str, Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def logits(self) -> Tensor:
        if isinstance(self.output, Tensor):
            return self.output
        value = (self.output.get("logits") if isinstance(self.output, dict)
                 else getattr(self.output, "logits", None))
        if not isinstance(value, Tensor):
            raise TypeError("Model output must contain tensor logits, or be a tensor.")
        return value
