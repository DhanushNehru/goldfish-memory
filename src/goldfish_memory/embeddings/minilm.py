"""Default embedder: all-MiniLM-L6-v2 (quantized ONNX) running on CPU.

Weights ship inside the package (Apache-2.0, see NOTICE), so the default
path works offline immediately after ``pip install``. Point ``model_dir``
at your own copy of compatible weights to swap models; nothing is ever
downloaded at runtime.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import Embedder, ModelNotProvisionedError

_BUNDLED = "_weights/all-MiniLM-L6-v2-quantized"


class MiniLMOnnxEmbedder(Embedder):
    """Sentence embedding via a local quantized all-MiniLM-L6-v2 ONNX model.

    Args:
        model_dir: Directory containing ``model.onnx`` and ``tokenizer.json``.
            When ``None``, the weights bundled with the package are used.

    Raises:
        ModelNotProvisionedError: if the required files are absent.
    """

    dimensions = 384

    def __init__(self, model_dir: str | Path | None = None) -> None:
        if model_dir is None:
            model_dir = Path(str(resources.files("goldfish_memory").joinpath(_BUNDLED)))
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        tokenizer_path = model_dir / "tokenizer.json"
        missing = [str(p) for p in (model_path, tokenizer_path) if not p.exists()]
        if missing:
            raise ModelNotProvisionedError(
                "Embedding model weights not found: "
                + ", ".join(missing)
                + ". goldfish-memory never downloads models. Reinstall the package to restore "
                "the bundled weights, or provision compatible all-MiniLM-L6-v2 ONNX weights "
                "yourself and set model_dir (or GOLDFISH_MODEL_DIR) to their directory."
            )
        import onnxruntime as ort
        from tokenizers import Tokenizer  # local import: keeps CLI startup fast

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=256)
        # Local file only: onnxruntime has no network path for file-based sessions.
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts into 384-dim unit vectors via mean pooling + L2 norm."""
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        enc = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed: dict[str, NDArray[np.int64]] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return cast(NDArray[np.float32], pooled / np.maximum(norms, 1e-9))
