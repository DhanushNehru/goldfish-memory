"""SQLite storage layer with a sqlite-vec vector index.

Layout: a ``memories`` table holds text and metadata; a ``memory_vec``
virtual table (sqlite-vec ``vec0``) holds the embedding for each memory,
keyed by the same rowid. Both live in one ordinary SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
from numpy.typing import NDArray

from .models import Memory

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'note',
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    session_id TEXT,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS compaction_log (
    id INTEGER PRIMARY KEY,
    summary_id INTEGER NOT NULL,
    child_id INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""

DEFAULT_DIR = ".goldfish"
DEFAULT_DB = "memory.db"


class Database:
    """Thin wrapper around the SQLite file holding memories and their vectors."""

    def __init__(self, path: str | Path, dimensions: int = 384) -> None:
        """Open (creating if needed) the store at ``path``.

        Args:
            path: Database file path, or ``":memory:"`` for an ephemeral store.
            dimensions: Embedding size; must match the embedder in use.
        """
        self.path = str(path)
        self.dimensions = dimensions
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._init_schema()

    @staticmethod
    def default_path(cwd: str | Path | None = None) -> Path:
        """Return the default store path: ``<cwd>/.goldfish/memory.db``."""
        return Path(cwd or Path.cwd()) / DEFAULT_DIR / DEFAULT_DB

    def _init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
            f"embedding float[{self.dimensions}])"
        )
        self.conn.commit()

    def add(self, memory: Memory, embedding: NDArray[np.float32]) -> int:
        """Insert a memory and its embedding; returns the new memory id."""
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO memories (kind, content, metadata, session_id, archived, pinned,"
            " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.kind,
                memory.content,
                json.dumps(memory.metadata),
                memory.session_id,
                int(memory.archived),
                int(memory.pinned),
                memory.created_at,
                now,
            ),
        )
        rowid = int(cur.lastrowid)  # type: ignore[arg-type]
        self.conn.execute(
            "INSERT INTO memory_vec(rowid, embedding) VALUES (?, ?)",
            (rowid, embedding.astype(np.float32).tobytes()),
        )
        self.conn.commit()
        return rowid

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            metadata=json.loads(row["metadata"]),
            session_id=row["session_id"],
            archived=bool(row["archived"]),
            pinned=bool(row["pinned"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, memory_id: int) -> Memory | None:
        """Fetch one memory by id, or ``None`` if it does not exist."""
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def recent(self, limit: int = 20, include_archived: bool = False) -> list[Memory]:
        """Return the newest memories, archived ones excluded unless requested."""
        sql = "SELECT * FROM memories"
        if not include_archived:
            sql += " WHERE archived = 0"
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        return [self._row_to_memory(r) for r in self.conn.execute(sql, (limit,)).fetchall()]

    def search_vec(
        self,
        embedding: NDArray[np.float32],
        k: int,
        include_archived: bool = False,
        kind: str | None = None,
        session_id: str | None = None,
        since: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """Vector search over embeddings, joined back to full memories.

        sqlite-vec returns squared-L2 distances; stored vectors are L2-normalized,
        so cosine similarity is ``1 - distance**2 / 2``.
        """
        filters: list[str] = []
        params: list[Any] = []
        if not include_archived:
            filters.append("archived = 0")
        if kind is not None:
            filters.append("kind = ?")
            params.append(kind)
        if session_id is not None:
            filters.append("session_id = ?")
            params.append(session_id)
        if since is not None:
            filters.append("created_at >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        # Oversample so post-join filters still leave k survivors.
        rows = self.conn.execute(
            """
            SELECT m.*, v.distance FROM memories m
            JOIN (
                SELECT rowid, distance FROM memory_vec
                WHERE embedding MATCH ? AND k = ?
            ) v ON v.rowid = m.id
            """
            + where
            + " ORDER BY v.distance LIMIT ?",
            (embedding.astype(np.float32).tobytes(), max(k * 4, 16), *params, k),
        ).fetchall()
        return [(self._row_to_memory(r), float(1.0 - r["distance"] ** 2 / 2.0)) for r in rows]

    def get_embeddings(self, ids: Iterable[int]) -> dict[int, NDArray[np.float32]]:
        """Return ``{id: embedding}`` for the given memory ids."""
        out: dict[int, NDArray[np.float32]] = {}
        for chunk in _chunks(list(ids), 500):
            marks = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT rowid, embedding FROM memory_vec WHERE rowid IN ({marks})", chunk
            ).fetchall()
            for r in rows:
                out[r["rowid"]] = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        return out

    def unarchived_older_than(self, cutoff: float, exclude_pinned: bool = True) -> list[Memory]:
        """List memories eligible for compaction: unarchived and older than ``cutoff``."""
        sql = "SELECT * FROM memories WHERE archived = 0 AND created_at < ?"
        params: list[Any] = [cutoff]
        if exclude_pinned:
            sql += " AND pinned = 0"
        sql += " AND kind != 'summary' ORDER BY created_at"
        return [self._row_to_memory(r) for r in self.conn.execute(sql, params).fetchall()]

    def set_archived(self, ids: Iterable[int], archived: bool = True) -> None:
        """Archive or unarchive memories; archived ones disappear from search."""
        now = time.time()
        self.conn.executemany(
            "UPDATE memories SET archived = ?, updated_at = ? WHERE id = ?",
            [(int(archived), now, i) for i in ids],
        )
        self.conn.commit()

    def set_pinned(self, memory_id: int, pinned: bool = True) -> None:
        """Pin or unpin a memory; pinned ones survive automatic compaction."""
        self.conn.execute(
            "UPDATE memories SET pinned = ?, updated_at = ? WHERE id = ?",
            (int(pinned), time.time(), memory_id),
        )
        self.conn.commit()

    def delete(self, memory_id: int) -> bool:
        """Permanently remove a memory and its vector. Returns True if it existed."""
        cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (memory_id,))
        self.conn.execute(
            "DELETE FROM compaction_log WHERE child_id = ? OR summary_id = ?",
            (memory_id, memory_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def count(self, include_archived: bool = False) -> int:
        """Number of stored memories."""
        sql = "SELECT COUNT(*) c FROM memories" + (
            "" if include_archived else " WHERE archived = 0"
        )
        return int(self.conn.execute(sql).fetchone()["c"])

    def meta_get(self, key: str) -> str | None:
        """Read a bookkeeping value (e.g. last compaction time)."""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def meta_set(self, key: str, value: str) -> None:
        """Write a bookkeeping value."""
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def log_compaction(self, summary_id: int, child_ids: Iterable[int]) -> None:
        """Record which memories a compaction summary replaced."""
        now = time.time()
        self.conn.executemany(
            "INSERT INTO compaction_log(summary_id, child_id, created_at) VALUES (?, ?, ?)",
            [(summary_id, c, now) for c in child_ids],
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.conn.close()


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
