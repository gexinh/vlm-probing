"""Common interface; category contracts live in each family's base.py."""
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .types import ProbeResult


class BaseMethod(ABC):
    """A reusable algorithm. Loading a model is the adapter's responsibility."""

    family: ClassVar[str]
    name: ClassVar[str]

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> ProbeResult:
        """Execute this method and return tensors with explicit metadata."""
