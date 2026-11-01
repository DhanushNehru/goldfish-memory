"""Data model for a single memory and search results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Memory:
    """One captured unit of context: a note, event, decision, or compaction summary.

    Attributes:
        id: Row id in the store; ``None`` before the memory is saved.
        kind: Free-form category, e.g. ``note``, ``decision``, ``error``, ``summary``.
        content: The remembered text.
        metadata: Arbitrary JSON-serializable extras (file paths, symbols, links).
        session_id: Optional id of the agent session that produced this memory.
        archived: Archived memories are excluded from search but kept on disk.
        pinned: Pinned memories are never archived by automatic compaction.
        created_at: Unix timestamp of capture.
        updated_at: Unix timestamp of the last mutation.
    """

    content: str
    kind: str = "note"
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    id: int | None = None
    archived: bool = False
    pinned: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """A memory returned by semantic search, with its similarity score.

    Attributes:
        memory: The matched memory.
        score: Cosine similarity in ``[-1, 1]``; higher means more relevant.
    """

    memory: Memory
    score: float
