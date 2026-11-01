"""Offline guarantee: the full workflow runs with all sockets disabled.

pytest-socket (enabled globally via addopts) blocks Python-level network in
every test run. CI additionally executes this module inside an empty network
namespace (``sudo unshare -n``) so native code paths (ONNX Runtime, the
tokenizer) cannot reach the network either. No fixtures here may provision
or download anything.
"""

import pytest
import pytest_socket

from goldfish_memory import MemoryEngine

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _sockets_off():
    pytest_socket.disable_socket()


def test_full_workflow_offline(tmp_path):
    eng = MemoryEngine(db_path=tmp_path / "memory.db")
    for i in range(6):
        eng.capture(f"deployment pipeline stage {i} builds the docker image and pushes it")
    hits = eng.search("how do we build containers", k=3)
    assert hits and "docker" in hits[0].memory.content
    block = eng.inject("add a caching stage to the pipeline", k=3)
    assert block.items
    eng.db.conn.execute("UPDATE memories SET created_at = 0")
    eng.db.conn.commit()
    result = eng.compact(min_age_days=14)
    assert result.summaries_created >= 1
    eng.close()


def test_missing_weights_error_is_offline_safe(tmp_path):
    from goldfish_memory.embeddings import MiniLMOnnxEmbedder, ModelNotProvisionedError

    with pytest.raises(ModelNotProvisionedError):
        MiniLMOnnxEmbedder(model_dir=tmp_path / "none")
