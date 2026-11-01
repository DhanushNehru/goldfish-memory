"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class ModelNotProvisionedError(RuntimeError):
    """Raised when local model weights are missing.

    The engine never downloads anything; this error tells the user exactly
    which files to place where, or how to restore the bundled weights.
    """


class Embedder(ABC):
    """Turns text into L2-normalized embedding vectors, fully offline."""

    dimensions: int

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed a batch of texts.

        Args:
            texts: One or more strings.

        Returns:
            ``float32`` array of shape ``(len(texts), dimensions)`` with
            unit-norm rows.
        """
