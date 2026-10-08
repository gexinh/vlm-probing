"""Three method families: causal interventions, attention analysis, and lenses."""
from .core import BaseMethod, ForwardTrace, ProbeResult
from .adapters import BaseModelAdapter, HookPoint, ModelReadout, TorchModelAdapter
from .adapters import CapabilityError, HeadSite, ModelSpec, ProbeInputs, TokenLayout, register_adapter
from .prober import Prober
from .api.lenses import LensMethods
from .api.attention import AttentionMethods
from .api.causal import CausalMethods
from .metrics import TokenMargin, SequenceLogProb, ClassScore
from . import visualization

__version__ = "0.8.0"
__all__ = ["BaseMethod", "ProbeResult", "ForwardTrace", "BaseModelAdapter",
           "HookPoint", "ModelReadout", "TorchModelAdapter", "Prober", "ModelSpec",
           "ProbeInputs", "TokenLayout", "HeadSite", "CapabilityError", "register_adapter",
           "LensMethods", "AttentionMethods", "CausalMethods", "TokenMargin", "SequenceLogProb", "ClassScore",
           "visualization"]
