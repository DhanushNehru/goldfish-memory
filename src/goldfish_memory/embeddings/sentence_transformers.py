"""Optional adapter for any locally cached sentence-transformers model.

Requires the ``sentence-transformers`` extra (``pip install
goldfish-memory[sentence-transformers]``) and a model already present in
the local cache. ``local_files_only=True`` is enforced: this adapter will
never touch the network, and fails fast if the model is not cached.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Embedder, ModelNotProvisionedError


class SentenceTransformersEmbedder(Embedder):
    """Use a sentence-transformers model from the local cache only.

    Args:
        model_name: Any sentence-transformers model id or local path.
        **kwargs: Forwarded to ``SentenceTransformer``.

    Raises:
        ModelNotProvisionedError: if the model is not available locally.
    """

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: object
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ModelNotProvisionedError(
                "sentence-transformers is not installed; "
                "install goldfish-memory[sentence-transformers] to use this adapter."
            ) from exc
        try:
            self._model = SentenceTransformer(
                model_name, device="cpu", local_files_only=True, **kwargs
            )
        except Exception as exc:  # model absent from local cache
            raise ModelNotProvisionedError(
                f"Model {model_name!r} is not in the local cache and this adapter never "
                "downloads. Fetch it yourself, then retry."
            ) from exc
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts into unit-norm vectors."""
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        out = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float32)
