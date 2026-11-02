"""Automatic compaction: cluster old memories, summarize, archive.

Everything runs locally on the stored embeddings. For each tight cluster of
old memories, an extractive summary (the most representative sentences,
chosen by similarity to the cluster centroid) replaces the originals in
search; the originals stay archived on disk and are linked in the
``compaction_log`` table, so compaction is auditable and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .embeddings.base import Embedder
from .models import Memory


@dataclass
class CompactionSummary:
    """One planned summary and the memories it replaces."""

    text: str
    child_ids: list[int]


def _sentences(text: str) -> list[str]:
    """Split text into rough sentences without any NLP dependency."""
    out: list[str] = []
    for chunk in text.replace("\n", " ").replace("!", ".").replace("?", ".").split("."):
        s = chunk.strip()
        if len(s) >= 8:
            out.append(s)
    return out or ([text.strip()] if text.strip() else [])


def cluster_memories(
    embeddings: dict[int, NDArray[np.float32]],
    similarity: float = 0.85,
) -> list[list[int]]:
    """Greedily group memory ids whose embeddings exceed ``similarity``.

    A memory joins the first cluster whose centroid is similar enough;
    otherwise it starts a new cluster. Single-pass, deterministic in the
    input order (callers pass memories oldest-first).
    """
    centroids: list[NDArray[np.float32]] = []
    clusters: list[list[int]] = []
    for mid, vec in embeddings.items():
        best, best_sim = -1, similarity
        for i, c in enumerate(centroids):
            sim = float(vec @ c)
            if sim >= best_sim:
                best, best_sim = i, sim
        if best < 0:
            clusters.append([mid])
            centroids.append(vec.copy())
        else:
            clusters[best].append(mid)
            c = centroids[best]
            centroids[best] = (c + (vec - c) / len(clusters[best])).astype(np.float32)
            centroids[best] /= np.linalg.norm(centroids[best])
    return clusters


def summarize_cluster(
    cluster: list[Memory],
    embedder: Embedder,
    max_sentences: int = 6,
) -> str:
    """Build an extractive summary for one cluster of related memories.

    Sentences from all members are ranked by similarity to the cluster
    centroid; the top few (near-duplicates dropped) are kept in
    chronological order.
    """
    centroid = np.mean([embedder.embed([m.content])[0] for m in cluster], axis=0).astype(np.float32)
    centroid /= np.linalg.norm(centroid)
    candidates: list[tuple[float, float, str]] = []  # (score, created_at, sentence)
    for m in cluster:
        for s in _sentences(m.content):
            vec = embedder.embed([s])[0]
            candidates.append((float(vec @ centroid), m.created_at, s))
    candidates.sort(key=lambda c: -c[0])
    chosen: list[tuple[float, str]] = []
    chosen_vecs: list[NDArray[np.float32]] = []
    for _score, ts, s in candidates:
        vec = embedder.embed([s])[0]
        if any(float(vec @ cv) > 0.95 for cv in chosen_vecs):
            continue  # near-duplicate of an already chosen sentence
        chosen.append((ts, s))
        chosen_vecs.append(vec)
        if len(chosen) >= max_sentences:
            break
    chosen.sort(key=lambda c: c[0])
    return " ".join(s.rstrip(".") + "." for _, s in chosen)


def plan_compaction(
    memories: list[Memory],
    embeddings: dict[int, NDArray[np.float32]],
    embedder: Embedder,
    min_cluster: int = 4,
    similarity: float = 0.85,
) -> list[CompactionSummary]:
    """Decide what to compact: clusters of at least ``min_cluster`` memories."""
    plans: list[CompactionSummary] = []
    for cluster_ids in cluster_memories(embeddings, similarity):
        if len(cluster_ids) < min_cluster:
            continue
        members = [m for m in memories if m.id in cluster_ids]
        text = summarize_cluster(members, embedder)
        plans.append(
            CompactionSummary(text=text, child_ids=[m.id for m in members if m.id is not None])
        )
    return plans
