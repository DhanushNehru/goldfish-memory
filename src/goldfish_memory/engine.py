"""MemoryEngine: capture, search, inject, compact - the public API surface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compaction import plan_compaction
from .config import Config
from .db import Database
from .embeddings.base import Embedder
from .models import Memory, SearchResult

_CHARS_PER_TOKEN = 4  # conservative estimate for budgeting injected context


@dataclass
class InjectionBlock:
    """Formatted memories ready to paste into an agent's next prompt.

    Attributes:
        text: Markdown block to inject.
        items: The memories included, most relevant first.
        token_estimate: Rough token count of ``text``.
    """

    text: str
    items: list[SearchResult]
    token_estimate: int


@dataclass
class CompactionResult:
    """Outcome of a compaction run."""

    summaries_created: int
    memories_archived: int


class MemoryEngine:
    """Persistent, searchable, on-device memory for a coding agent.

    Args:
        db_path: SQLite file for this project (default ``.goldfish/memory.db``
            under the current directory, or ``$GOLDFISH_DB``).
        embedder: Embedding provider; defaults to the bundled MiniLM ONNX
            model. Any :class:`~goldfish_memory.embeddings.Embedder` works.
        config: Behavior settings; defaults to ``config.json`` next to the
            database, or built-in defaults.

    All work happens on the local machine; the engine performs no network
    access of any kind.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        embedder: Embedder | None = None,
        config: Config | None = None,
    ) -> None:
        import os

        if db_path is None:
            db_path = os.environ.get("GOLDFISH_DB") or Database.default_path()
        self._db_path = db_path
        self.config = config or Config.load(db_path)
        self._embedder = embedder
        self._db: Database | None = None

    @property
    def db(self) -> Database:
        """Lazily opened store (so ``--help`` needs no files or model)."""
        if self._db is None:
            self._db = Database(self._db_path, dimensions=self.embedder.dimensions)
        return self._db

    @property
    def embedder(self) -> Embedder:
        """Lazily constructed embedding provider."""
        if self._embedder is None:
            from .embeddings.minilm import MiniLMOnnxEmbedder

            self._embedder = MiniLMOnnxEmbedder(self.config.model_dir)
        return self._embedder

    def capture(
        self,
        content: str,
        kind: str = "note",
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        pinned: bool = False,
    ) -> Memory:
        """Store a new memory and index it for semantic search.

        Args:
            content: The text to remember.
            kind: Category such as ``note``, ``decision``, ``error``.
            metadata: Optional JSON-serializable extras.
            session_id: Optional agent session this belongs to.
            pinned: Pinned memories are never auto-compacted.

        Returns:
            The stored memory with its id filled in.
        """
        memory = Memory(
            content=content.strip(),
            kind=kind,
            metadata=metadata or {},
            session_id=session_id,
            pinned=pinned,
        )
        memory.id = self.db.add(memory, self.embedder.embed([memory.content])[0])
        self._maybe_auto_compact()
        return memory

    def search(
        self,
        query: str,
        k: int = 5,
        kind: str | None = None,
        session_id: str | None = None,
        since_days: float | None = None,
        include_archived: bool = False,
    ) -> list[SearchResult]:
        """Semantic search over captured memories, most relevant first."""
        since = time.time() - since_days * 86400 if since_days is not None else None
        hits = self.db.search_vec(
            self.embedder.embed([query])[0],
            k,
            include_archived=include_archived,
            kind=kind,
            session_id=session_id,
            since=since,
        )
        return [SearchResult(memory=m, score=s) for m, s in hits]

    def inject(self, task: str, k: int = 6, max_tokens: int = 800) -> InjectionBlock:
        """Build the context block for the agent's next session.

        Args:
            task: What the agent is about to do; used as the search query.
            k: Maximum memories to include.
            max_tokens: Budget for the rendered block.

        Returns:
            An :class:`InjectionBlock` with Markdown ready to prepend to a prompt.
        """
        results = self.search(task, k=k)
        lines = ["## Relevant memory (goldfish-memory)"]
        items: list[SearchResult] = []
        budget = max_tokens * _CHARS_PER_TOKEN - len(lines[0])
        for r in results:
            date = time.strftime("%Y-%m-%d", time.localtime(r.memory.created_at))
            line = f"- [{r.memory.kind}, {date}] {r.memory.content}"
            if budget - len(line) < 0:
                if not items and budget > 24:
                    # Always surface the top hit, truncated, instead of
                    # wrongly reporting that no memories exist.
                    lines.append(line[: budget - 1] + "...")
                    items.append(r)
                break
            lines.append(line)
            items.append(r)
            budget -= len(line)
        if not items:
            lines.append("- (no relevant memories yet)")
        text = "\n".join(lines)
        return InjectionBlock(text=text, items=items, token_estimate=len(text) // _CHARS_PER_TOKEN)

    def compact(
        self,
        min_age_days: float | None = None,
        min_cluster: int = 4,
        dry_run: bool = False,
    ) -> CompactionResult:
        """Summarize and archive clusters of old memories.

        Extractive summaries stay searchable as ``kind="summary"``; originals
        are archived (kept on disk, excluded from search) and linked in the
        compaction log. Pinned and previously summarized memories are skipped.
        """
        age = self.config.compact_min_age_days if min_age_days is None else min_age_days
        cutoff = time.time() - age * 86400
        old = self.db.unarchived_older_than(cutoff)
        if not old:
            return CompactionResult(0, 0)
        ids = [m.id for m in old if m.id is not None]
        embeddings = self.db.get_embeddings(ids)
        plans = plan_compaction(old, embeddings, self.embedder, min_cluster=min_cluster)
        if dry_run:
            return CompactionResult(len(plans), sum(len(p.child_ids) for p in plans))
        archived = 0
        for plan in plans:
            summary = Memory(
                content=plan.text,
                kind="summary",
                metadata={"compacted": plan.child_ids},
            )
            summary.id = self.db.add(summary, self.embedder.embed([summary.content])[0])
            self.db.set_archived(plan.child_ids, True)
            self.db.log_compaction(summary.id, plan.child_ids)
            archived += len(plan.child_ids)
        self.db.meta_set("last_compaction", str(time.time()))
        return CompactionResult(len(plans), archived)

    def _maybe_auto_compact(self) -> None:
        """Trigger compaction during capture when enough old memories pile up."""
        cfg = self.config
        if not cfg.auto_compact:
            return
        last = float(self.db.meta_get("last_compaction") or 0)
        if time.time() - last < cfg.compact_interval_hours * 3600:
            return
        cutoff = time.time() - cfg.compact_min_age_days * 86400
        if len(self.db.unarchived_older_than(cutoff)) >= cfg.compact_batch:
            self.compact()

    def recent(self, limit: int = 20, include_archived: bool = False) -> list[Memory]:
        """List the newest memories."""
        return self.db.recent(limit, include_archived)

    def pin(self, memory_id: int, pinned: bool = True) -> None:
        """Protect a memory from automatic compaction (or unpin it)."""
        self.db.set_pinned(memory_id, pinned)

    def forget(self, memory_id: int) -> bool:
        """Permanently delete one memory. Returns True if it existed."""
        return self.db.delete(memory_id)

    def stats(self) -> dict[str, int | str]:
        """Basic store statistics: counts and database location."""
        return {
            "db_path": str(self._db_path),
            "active": self.db.count(),
            "total_including_archived": self.db.count(include_archived=True),
            "embedder": type(self.embedder).__name__,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> MemoryEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
