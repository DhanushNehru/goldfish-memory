"""Smoke test: the benchmark runs end to end on a tiny store."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark", Path(__file__).parent.parent / "scripts" / "benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_smoke(tmp_path):
    from goldfish_memory import MemoryEngine

    engine = MemoryEngine(db_path=tmp_path / "b.db")
    topic_ids = benchmark.build_store(engine, 80)
    assert sum(len(v) for v in topic_ids.values()) == 88  # 80 filler + 8 needles
    queries = list(benchmark.QUERIES.values())
    index_ms = benchmark.bench_index_latency(engine, queries, 3)
    brute_ms = benchmark.bench_bruteforce_latency(engine, queries, 3)
    assert index_ms > 0 and brute_ms > 0
    costs = benchmark.context_costs(engine, queries, 3, 400)
    assert 0.0 <= costs["budget_recall_inject"] <= 1.0
    assert costs["full_history_tokens"] > costs["inject_tokens_mean"]
    engine.close()
