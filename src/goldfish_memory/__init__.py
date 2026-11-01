"""goldfish-memory: on-device persistent memory for AI coding agents.

Capture decisions, errors, and notes; search them semantically; inject the
relevant ones into the next session's prompt; let compaction keep the store
small. Everything runs locally in SQLite with a bundled embedding model -
the engine makes no network calls, ever.

Quick start::

    from goldfish_memory import MemoryEngine

    with MemoryEngine() as engine:
        engine.capture("We chose SQLite for the store.", kind="decision")
        print(engine.inject("what storage are we using?").text)
"""

from .config import Config
from .embeddings import Embedder, MiniLMOnnxEmbedder, ModelNotProvisionedError
from .engine import CompactionResult, InjectionBlock, MemoryEngine
from .models import Memory, SearchResult

__version__ = "0.1.0"
__all__ = [
    "CompactionResult",
    "Config",
    "Embedder",
    "InjectionBlock",
    "Memory",
    "MemoryEngine",
    "MiniLMOnnxEmbedder",
    "ModelNotProvisionedError",
    "SearchResult",
    "__version__",
]
