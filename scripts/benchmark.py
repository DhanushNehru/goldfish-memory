#!/usr/bin/env python3
"""Benchmark goldfish-memory against the naive full-history approach.

Compares, on a synthetic store of N memories (deterministic seed):

* retrieval latency: sqlite-vec index search vs brute-force rescan of every
  embedding in the store (what "just embed everything at query time" costs);
* context cost: tokens needed to hand the agent the top-k injected block vs
  the entire raw history;
* budget recall: whether the truly relevant memory fits inside a fixed
  token budget under each approach.

Runs fully offline on the bundled model. Usage:

    python scripts/benchmark.py [--memories 5000] [--queries 100] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from goldfish_memory import MemoryEngine

TOPICS = {
    "database": "schema migration index query transaction table postgres sqlite",
    "frontend": "css layout flexbox component button render responsive grid",
    "auth": "login token jwt session cookie oauth refresh password",
    "deploy": "docker image kubernetes rollout pipeline build registry",
    "testing": "pytest fixture mock coverage flaky assertion integration",
    "billing": "stripe invoice webhook subscription payment refund plan",
    "cache": "redis eviction ttl cache invalidation memory key",
    "api": "endpoint route handler rest graphql request response status",
}

# One unique, fact-bearing "needle" memory per topic, buried as the OLDEST
# entries in the store. Each query asks for exactly that fact. This is what
# real recall looks like; the repetitive filler is just the haystack.
NEEDLES = {
    "database": "migration safety rule: always run schema migrations "
    "inside a transaction with a tested rollback",
    "frontend": "mobile layout fix: the sidebar collapses below 768px using the useMediaQuery hook",
    "auth": "refresh tokens live in httpOnly cookies and rotate every 24 hours",
    "deploy": "release pipeline fact: canary deploys go to 5 percent of traffic for 30 minutes",
    "testing": "flaky test fact: test_checkout_total is order-dependent "
    "because of the shared cart fixture",
    "billing": "payment retry rule: failed invoices retry at 3, 5, and 7 days before cancellation",
    "cache": "cache expiry fact: session cache entries expire after 900 seconds of idle time",
    "api": "status code rule: the export endpoint returns 202 with a job id while processing",
}

QUERIES = {
    "database": "what is the rule for running schema migrations safely",
    "frontend": "how is the broken mobile sidebar layout fixed",
    "auth": "where do refresh tokens live and how often do they rotate",
    "deploy": "how much traffic do canary deploys get and for how long",
    "testing": "which test is order-dependent and why",
    "billing": "when are failed invoice payments retried",
    "cache": "after how many seconds do idle session cache entries expire",
    "api": "what status code does the export endpoint return while processing",
}

CHARS_PER_TOKEN = 4


def build_store(engine: MemoryEngine, n: int, seed: int = 0) -> dict[str, list[int]]:
    """Insert needle memories, then n filler memories across topics."""
    rng = np.random.default_rng(seed)
    topic_ids: dict[str, list[int]] = {t: [] for t in TOPICS}
    names = list(TOPICS)
    texts: list[str] = [f"{t} note: {fact}" for t, fact in NEEDLES.items()]
    topics: list[str] = list(NEEDLES)
    for i in range(n):
        topic = names[i % len(names)]
        words = TOPICS[topic].split()
        pick = rng.choice(words, size=8, replace=True)
        filler = " ".join(rng.choice(words, size=3))
        texts.append(f"{topic} note {i}: {' '.join(pick)} {filler}")
        topics.append(topic)
    # Bulk path: embed in batches and write via the storage layer, the same
    # way an agent backfilling a transcript would.
    from goldfish_memory.models import Memory

    total = len(texts)
    for start in range(0, total, 256):
        chunk = texts[start : start + 256]
        vectors = engine.embedder.embed(chunk)
        for text, topic, vec in zip(chunk, topics[start : start + 256], vectors, strict=True):
            mid = engine.db.add(Memory(content=text, kind="note"), vec)
            topic_ids[topic].append(mid)
    return topic_ids


def bench_index_latency(engine: MemoryEngine, queries: list[str], k: int) -> float:
    """Mean ms per query through the sqlite-vec index."""
    start = time.perf_counter()
    for q in queries:
        engine.search(q, k=k)
    return (time.perf_counter() - start) / len(queries) * 1000


def bench_bruteforce_latency(engine: MemoryEngine, queries: list[str], k: int) -> float:
    """Mean ms per query rescanning every stored embedding with numpy."""
    all_ids = [m.id for m in engine.recent(10**9, include_archived=True)]
    matrix = np.stack([engine.db.get_embeddings(all_ids)[i] for i in all_ids])
    start = time.perf_counter()
    for q in queries:
        qv = engine.embedder.embed([q])[0]
        sims = matrix @ qv
        np.argpartition(-sims, k)[:k]
    return (time.perf_counter() - start) / len(queries) * 1000


def context_costs(
    engine: MemoryEngine, queries: list[str], k: int, budget: int
) -> dict[str, float]:
    """Token cost of injected block vs full history, and budget recall."""
    full_history = engine.recent(10**9, include_archived=True)
    full_tokens = sum(len(m.content) for m in full_history) // CHARS_PER_TOKEN
    inject_tokens: list[int] = []
    recall_inject = 0
    recall_full = 0
    for topic, q in QUERIES.items():
        needle = NEEDLES[topic]
        block = engine.inject(q, k=k, max_tokens=budget)
        inject_tokens.append(block.token_estimate)
        if any(needle in r.memory.content for r in block.items):
            recall_inject += 1
        # Naive: history dumped newest-first until the budget runs out.
        spent = 0
        for m in full_history:
            spent += len(m.content) // CHARS_PER_TOKEN
            if spent > budget:
                break
            if needle in m.content:
                recall_full += 1
                break
    n = len(QUERIES)
    return {
        "inject_tokens_mean": float(np.mean(inject_tokens)),
        "full_history_tokens": float(full_tokens),
        "budget_recall_inject": recall_inject / n,
        "budget_recall_full_history": recall_full / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memories", type=int, default=5000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--budget", type=int, default=800, help="token budget for injection")
    parser.add_argument("--json", type=Path, help="also write results as JSON")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        engine = MemoryEngine(db_path=Path(tmp) / "bench.db")
        print(f"building synthetic store of {args.memories} memories (offline)...")
        build_store(engine, args.memories)
        queries = [q for _ in range(args.queries // len(QUERIES) + 1) for q in QUERIES.values()]
        queries = queries[: args.queries]

        index_ms = bench_index_latency(engine, queries, args.k)
        brute_ms = bench_bruteforce_latency(engine, queries, args.k)
        costs = context_costs(engine, queries, args.k, args.budget)
        engine.close()

    results = {
        "memories": args.memories,
        "queries": args.queries,
        "k": args.k,
        "index_latency_ms": round(index_ms, 2),
        "bruteforce_latency_ms": round(brute_ms, 2),
        "speedup": round(brute_ms / index_ms, 1) if index_ms else None,
        **{k2: round(v, 3) if isinstance(v, float) else v for k2, v in costs.items()},
    }
    print(json.dumps(results, indent=2))
    print(
        "\n| Metric | goldfish-memory | naive full-history |\n"
        "|---|---|---|\n"
        f"| Retrieval latency / query | {results['index_latency_ms']} ms "
        f"| {results['bruteforce_latency_ms']} ms |\n"
        f"| Tokens handed to the agent | ~{int(results['inject_tokens_mean'])} "
        f"| {int(results['full_history_tokens'])} |\n"
        f"| Buried fact included within {args.budget}-token context budget "
        f"| {results['budget_recall_inject']:.0%} | {results['budget_recall_full_history']:.0%} |"
    )
    print(
        "\\nNote: the full history *contains* every fact - the last row measures "
        "whether the fact fits the token budget when history is dumped newest-first, "
        "not answer quality or general accuracy."
    )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
