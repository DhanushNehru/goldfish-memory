"""Pluggable embedding providers.

Every provider turns text into a fixed-size L2-normalized vector and runs
entirely on the local machine. The default is :class:`MiniLMOnnxEmbedder`
using the weights bundled with the package; alternates can be supplied by
passing any :class:`Embedder` implementation to the engine.
"""

from .base import Embedder, ModelNotProvisionedError
from .minilm import MiniLMOnnxEmbedder

__all__ = ["Embedder", "MiniLMOnnxEmbedder", "ModelNotProvisionedError"]
