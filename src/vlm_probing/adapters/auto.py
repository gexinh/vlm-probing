"""Small explicit adapter registry, with optional Hugging Face integration."""
from collections.abc import Callable

from torch import nn

from .spec import CapabilityError
from .torch import TorchModelAdapter

_FACTORIES: dict[type, Callable[[nn.Module], TorchModelAdapter]] = {}


def register_adapter(model_class: type, factory: Callable, *, replace: bool = False):
    """Register a factory for an exact Python model class; never guess subclasses."""
    if model_class in _FACTORIES and not replace:
        raise ValueError(f"adapter already registered for {model_class.__name__}")
    _FACTORIES[model_class] = factory


def auto_adapter(model):
    if type(model) in _FACTORIES:
        return _FACTORIES[type(model)](model)
    factory = getattr(model, "probing_adapter", None)
    if callable(factory):
        return factory()
    known = {
        ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
        ("transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
    }
    if (type(model).__module__, type(model).__name__) in known:
        from .huggingface import make_adapter
        return make_adapter(model)
    raise CapabilityError(
        f"No automatic adapter for {type(model).__name__}. Pass adapter=TorchModelAdapter(..., "
        "spec=ModelSpec(...)), register_adapter(ModelClass, factory), or implement probing_adapter()."
    )
