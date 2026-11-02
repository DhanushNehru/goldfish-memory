# Benchmarks

Run it yourself (fully offline once installed):

```bash
python scripts/benchmark.py --memories 2000 --queries 100
```

## What it measures

The comparison is against the naive full-history approach: keep everything
and hand the agent the whole history (or rescan every stored embedding at
query time).

The synthetic store holds 2,000 filler memories across 8 topics plus one
unique fact-bearing "needle" memory per topic, inserted **oldest**. Each
query asks for exactly one needle's fact. All data is synthetic and
generated with a fixed seed - no user content.

Metrics:

* **Retrieval latency** - mean ms per query: sqlite-vec index search vs a
  numpy brute-force rescan of all embeddings.
* **Tokens handed to the agent** - the injected top-k block vs the entire
  raw history (estimated at 4 chars/token).
* **Buried fact within budget** - whether the needle memory reaches the
  agent inside an 800-token context budget: selected by semantic retrieval,
  or included in a newest-first dump of raw history cut off at the budget.

## Latest run

Environment: GitHub-free x86_64 laptop-class CPU (sandbox container),
Python 3.10, quantized all-MiniLM-L6-v2 on CPU, 2,000 memories, 100 queries.

| Metric | goldfish-memory | Naive full-history |
|---|---|---|
| Retrieval latency / query | 13.1 ms | 11.9 ms |
| Tokens handed to the agent | ~189 | ~49,975 |
| Buried fact within 800-token budget | 100% (8/8) | 0% (0/8) |

## How to read this

* Latency is a wash at 2,000 memories: sqlite-vec's flat index and the numpy
  rescan are both exact scans. goldfish-memory's win is not raw speed at
  this size - it is context economy.
* The full history *does* contain every fact. The 0% is not "the naive
  approach is inaccurate"; it is "50k tokens do not fit an 800-token (or an
  8k-token) budget", so the fact never reaches the prompt. Retrieval chooses
  the few memories that earn their tokens.
* This benchmark says nothing about downstream answer quality; it measures
  whether the right context arrives within budget.

A broader quality suite (labeled datasets, recall@k across embedding models,
CI regression gates) is tracked as a roadmap issue.
