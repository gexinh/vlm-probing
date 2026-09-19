"""Six lens algorithms sharing the BaseLens interface."""
from .base import BaseLens
from .logit import LogitLens
from .embed import EmbedLens
from .tuned import TunedLens
from .attention import AttentionLens
from .jacobian import JacobianLens
from .patchscope import Patchscope

__all__ = ["BaseLens", "LogitLens", "EmbedLens", "TunedLens", "AttentionLens", "JacobianLens", "Patchscope"]
