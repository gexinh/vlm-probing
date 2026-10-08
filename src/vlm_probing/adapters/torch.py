"""Explicit PyTorch module sites, without architecture-name heuristics.

This adapter accepts already processed model kwargs. It does not infer image
token positions or expose attention internals hidden inside fused kernels.
"""
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal
import copy

import torch
from torch import Tensor, nn

from .base import BaseModelAdapter
from ..core.types import ForwardTrace
from .spec import ModelSpec


@dataclass(frozen=True)
class HookPoint:
    """Container selector, followed by an optional disjoint last-axis slice."""
    module: str
    kind: Literal["input", "output"] = "output"
    selector: int | str | None = None
    tensor_slice: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"input", "output"}:
            raise ValueError("kind must be input or output")
        if self.kind == "input" and self.selector is None:
            raise ValueError("Input hooks require a positional index or keyword selector")
        if self.tensor_slice is not None:
            if (not isinstance(self.tensor_slice, tuple) or len(self.tensor_slice) != 2
                    or any(type(i) is not int for i in self.tensor_slice)
                    or not 0 <= self.tensor_slice[0] < self.tensor_slice[1]):
                raise ValueError("tensor_slice must be a (start, stop) pair with 0 <= start < stop")


class ModelReadout:
    """Apply final norm once, then output head and any model-specific transform.

    The caller owns the modules; this wrapper does not change their parameters,
    device, gradient settings, or training mode.
    """
    def __init__(self, head: nn.Module, *, norm: nn.Module | None = None,
                 input_normalized: bool = False,
                 output_transform: Callable[[Tensor], Tensor] | None = None):
        self.head, self.norm = head, norm
        self.input_normalized = input_normalized
        self.output_transform = output_transform

    def __call__(self, hidden: Tensor) -> Tensor:
        if self.norm is not None and not self.input_normalized:
            hidden = self.norm(hidden)
        logits = self.head(hidden)
        return self.output_transform(logits) if self.output_transform else logits


class TorchModelAdapter(BaseModelAdapter):
    """A site must execute once per run; repeated/missing sites raise an error.

    Captures are after interventions. Detached runs clone captured values;
    gradient runs retain the actual graph tensors. Concurrent or nested use of
    the same adapter is unsupported. Model eval flags and hooks are restored,
    including when an intervention or the model raises an exception.
    """
    def __init__(self, model: nn.Module, sites: Mapping[str, HookPoint], *,
                 model_id: str = "unspecified", readout: ModelReadout | None = None,
                 spec: ModelSpec | None = None):
        self.model, self.sites = model, dict(sites)
        self.model_id, self.readout = model_id, readout
        self.spec = spec
        if spec is not None:
            spec.validate(sites)
        self._active = False
        if len(set(self.sites.values())) != len(self.sites):
            raise ValueError("Duplicate aliases for the same hook site are ambiguous")
        for spec in self.sites.values():
            model.get_submodule(spec.module)  # Fail before any model execution.
        # Disjoint packed Q/K/V slices are legal; overlapping aliases are not.
        points = list(self.sites.values())
        for i, left in enumerate(points):
            for right in points[i + 1:]:
                if (left.module, left.kind, left.selector) != (right.module, right.kind, right.selector):
                    continue
                a, b = left.tensor_slice, right.tensor_slice
                if a is None or b is None or max(a[0], b[0]) < min(a[1], b[1]):
                    raise ValueError("Overlapping aliases for the same hook tensor are ambiguous")

    @staticmethod
    def _extract(value: Any, selector: int | str | None) -> Tensor:
        tensor = value if selector is None else value[selector]
        if not isinstance(tensor, Tensor):
            raise TypeError("The selected hook value is not a Tensor")
        return tensor

    @staticmethod
    def _replace(value: Any, selector: int | str | None, tensor: Tensor) -> Any:
        if selector is None:
            return tensor
        if isinstance(value, tuple):
            items = list(value)
            items[selector] = tensor
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        result = copy.copy(value)
        result[selector] = tensor
        return result

    def run(self, inputs: Mapping[str, Any], *, capture: Iterable[str] = (),
            interventions: Mapping[str, Callable[[Tensor], Tensor]] | None = None,
            grad: bool = False) -> ForwardTrace:
        if self._active:
            raise RuntimeError("Concurrent/nested execution on this adapter is unsupported")
        if grad and torch.is_inference_mode_enabled():
            raise ValueError("Gradient capture cannot run inside torch.inference_mode()")
        capture = tuple(capture)
        edits = dict(interventions or {})
        names = tuple(dict.fromkeys((*capture, *edits)))
        unknown = set(names) - self.sites.keys()
        if unknown:
            raise KeyError(f"Unknown hook sites: {sorted(unknown)}")
        if inputs.get("use_cache") is True or inputs.get("past_key_values") is not None:
            raise ValueError("This adapter supports uncached forward passes only")
        config = getattr(self.model, "config", None)
        if getattr(config, "use_cache", False) and inputs.get("use_cache") is not False:
            raise ValueError("Pass use_cache=False explicitly for models that enable caching by default")
        cache: dict[str, Tensor] = {}
        captured_versions: dict[str, int] = {}
        visits = dict.fromkeys(names, 0)
        handles = []
        states = [(module, module.training) for module in self.model.modules()]
        self._active = True

        def transform(name: str, tensor: Tensor) -> Tensor:
            visits[name] += 1
            if visits[name] > 1:
                raise RuntimeError(f"Site {name!r} executed more than once; specify a single-pass site")
            edited = edits[name](tensor) if name in edits else tensor
            if not isinstance(edited, Tensor):
                raise TypeError(f"Intervention at {name!r} must return a Tensor")
            if edited.shape != tensor.shape or edited.device != tensor.device or edited.dtype != tensor.dtype:
                raise ValueError(f"Intervention at {name!r} must preserve shape, device, and dtype")
            if name in capture:
                cache[name] = edited if grad else edited.detach().clone()
                if grad:
                    captured_versions[name] = edited._version
            return edited

        def output_hook(name: str, spec: HookPoint):
            def hook(module, args, kwargs, output):
                edited = edit_tensor(name, spec, self._extract(output, spec.selector))
                return self._replace(output, spec.selector, edited)
            return hook

        def input_hook(name: str, spec: HookPoint):
            def hook(module, args, kwargs):
                keyword = isinstance(spec.selector, str)
                value = kwargs if keyword else args
                edited = edit_tensor(name, spec, self._extract(value, spec.selector))
                replaced = self._replace(value, spec.selector, edited)
                return (args, replaced) if keyword else (replaced, kwargs)
            return hook

        def edit_tensor(name: str, spec: HookPoint, value: Tensor) -> Tensor:
            if spec.tensor_slice is None:
                return transform(name, value)
            start, stop = spec.tensor_slice
            if value.ndim < 1 or stop > value.shape[-1]:
                raise ValueError(f"tensor_slice at {name!r} exceeds the hook tensor width")
            edited = transform(name, value[..., start:stop])
            result = value.clone()
            result[..., start:stop] = edited
            return result

        try:
            self.model.eval()
            for name in names:
                spec = self.sites[name]
                module = self.model.get_submodule(spec.module)
                if spec.kind == "output":
                    handles.append(module.register_forward_hook(output_hook(name, spec), with_kwargs=True))
                else:
                    handles.append(module.register_forward_pre_hook(input_hook(name, spec), with_kwargs=True))
            with torch.enable_grad() if grad else torch.no_grad():
                output = self.model(**dict(inputs))
            missing = [name for name, count in visits.items() if count == 0]
            if missing:
                raise RuntimeError(f"Requested sites were not executed: {missing}")
            mutated = [name for name, version in captured_versions.items()
                       if cache[name]._version != version]
            if mutated:
                raise RuntimeError(f"Gradient captures were mutated in-place downstream: {mutated}")
            return ForwardTrace(output, cache, {"model_id": self.model_id,
                                "sites": list(names), "grad": grad,
                                "capture_stage": "after_intervention"})
        finally:
            for handle in handles:
                handle.remove()
            for module, training in states:
                module.training = training
            self._active = False
