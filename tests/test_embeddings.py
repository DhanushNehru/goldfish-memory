"""The real bundled model: shape, normalization, semantic ordering."""

import numpy as np
import pytest

from goldfish_memory.embeddings import MiniLMOnnxEmbedder, ModelNotProvisionedError


@pytest.fixture(scope="module")
def model():
    return MiniLMOnnxEmbedder()


def test_embed_shape_and_norm(model):
    vecs = model.embed(["hello world", "another sentence entirely"])
    assert vecs.shape == (2, 384) and vecs.dtype == np.float32
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_semantic_ordering(model):
    vecs = model.embed(
        [
            "we store agent memories in sqlite",
            "the memory store uses a sqlite database file",
            "recipe for chocolate chip cookies",
        ]
    )
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


def test_empty_batch(model):
    assert model.embed([]).shape == (0, 384)


def test_missing_model_dir_fails_fast(tmp_path):
    with pytest.raises(ModelNotProvisionedError, match="never downloads"):
        MiniLMOnnxEmbedder(model_dir=tmp_path / "empty")


def test_real_engine_end_to_end(real_engine):
    real_engine.capture("the billing service talks to stripe over webhooks", kind="note")
    real_engine.capture("sourdough needs a hot dutch oven", kind="note")
    hits = real_engine.search("how do payments get confirmed", k=2)
    assert "stripe" in hits[0].memory.content
