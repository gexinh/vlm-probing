"""Model execution contract shared by every method family."""
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from torch import Tensor

from ..core.types import ForwardTrace


class BaseModelAdapter(ABC):
    @abstractmethod
    def run(self, inputs: Mapping[str, Any], *, capture: Iterable[str] = (),
            interventions: Mapping[str, Callable[[Tensor], Tensor]] | None = None,
            grad: bool = False) -> ForwardTrace:
        """Run the real model with scoped capture and optional tensor edits."""
