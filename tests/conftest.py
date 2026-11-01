"""Shared fixtures: a deterministic toy embedder and engines built on it."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from goldfish_memory import Config, MemoryEngine
from goldfish_memory.embeddings.base import Embedder


class ToyEmbedder(Embedder):
    """Deterministic toy embedder for unit tests ONLY - not semantic.

    Each word maps to a seeded pseudo-random unit vector and a text's
    embedding is the normalized sum of its word vectors. Tests can control
    similarity by sharing or not sharing words. Never shipped; real-model
    behavior is covered by test_embeddings.py and the offline suite.
    """

    dimensions = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
                out[i] += np.random.default_rng(seed).standard_normal(self.dimensions)
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out


@pytest.fixture
def toy() -> ToyEmbedder:
    return ToyEmbedder()


@pytest.fixture
def engine(tmp_path, toy) -> MemoryEngine:
    eng = MemoryEngine(
        db_path=tmp_path / "memory.db",
        embedder=toy,
        config=Config(auto_compact=False),
    )
    yield eng
    eng.close()


@pytest.fixture
def real_engine(tmp_path) -> MemoryEngine:
    """Engine on the real bundled model (slower; for integration tests)."""
    eng = MemoryEngine(db_path=tmp_path / "memory.db", config=Config(auto_compact=False))
    yield eng
    eng.close()
