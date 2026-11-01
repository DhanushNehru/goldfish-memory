"""The ``goldfish`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from .db import Database
from .embeddings.base import ModelNotProvisionedError
from .engine import MemoryEngine

DEMO_NOTES = [
    ("decision", "Chose SQLite + sqlite-vec for storage so the whole memory store is one file."),
    ("decision", "Rejected a hosted vector DB: the engine must run fully offline."),
    (
        "error",
        "pytest-socket blocked urllib3 during the offline test; provision models beforehand.",
    ),
    ("note", "The API server caches embeddings in memory for the lifetime of the process."),
    ("decision", "CLI stays on argparse to keep the dependency footprint small."),
    ("note", "Benchmark compares vector search against a naive full-history scan."),
    ("error", "Fixed timezone bug: memory timestamps were naive datetimes, now Unix time."),
    (
        "note",
        "Compaction clusters old memories by cosine similarity and keeps extractive summaries.",
    ),
]


def _engine(args: argparse.Namespace) -> MemoryEngine:
    return MemoryEngine(db_path=args.db)


def _cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.db or Database.default_path())
    engine = MemoryEngine(db_path=path)
    engine.config.save(path)
    _ = engine.db  # create schema
    engine.close()
    print(f"Initialized goldfish-memory store at {path}")
    print('Try it: goldfish add "my first note" && goldfish search "note"')
    return 0


def _cmd_add(args: argparse.Namespace) -> int:
    text = args.text or sys.stdin.read().strip()
    if not text:
        print("nothing to capture", file=sys.stderr)
        return 2
    with _engine(args) as engine:
        m = engine.capture(text, kind=args.kind, session_id=args.session, pinned=args.pin)
        print(f"saved memory #{m.id} ({m.kind})")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        results = engine.search(args.query, k=args.k, kind=args.kind, session_id=args.session)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "id": r.memory.id,
                        "kind": r.memory.kind,
                        "score": round(r.score, 4),
                        "content": r.memory.content,
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        for r in results:
            print(f"[{r.score:+.3f}] #{r.memory.id} ({r.memory.kind}) {r.memory.content}")
        if not results:
            print("no matches")
    return 0


def _cmd_inject(args: argparse.Namespace) -> int:
    task = args.task or sys.stdin.read().strip()
    with _engine(args) as engine:
        block = engine.inject(task, k=args.k, max_tokens=args.max_tokens)
    print(block.text)
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        result = engine.compact(min_age_days=args.older_than, dry_run=args.dry_run)
    verb = "would create" if args.dry_run else "created"
    print(
        f"compaction {verb} {result.summaries_created} summaries, "
        f"archiving {result.memories_archived} memories"
    )
    return 0


def _cmd_recent(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        for m in engine.recent(args.n, include_archived=args.all):
            flag = " (archived)" if m.archived else ""
            date = time.strftime("%Y-%m-%d %H:%M", time.localtime(m.created_at))
            print(f"#{m.id} [{m.kind}] {date}{flag} {m.content}")
    return 0


def _cmd_pin(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        engine.pin(args.id, pinned=not args.unpin)
    print(("unpinned" if args.unpin else "pinned") + f" memory #{args.id}")
    return 0


def _cmd_forget(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        ok = engine.forget(args.id)
    print(f"deleted memory #{args.id}" if ok else f"memory #{args.id} not found")
    return 0 if ok else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    with _engine(args) as engine:
        for key, value in engine.stats().items():
            print(f"{key}: {value}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a self-contained tour on a throwaway store with synthetic data."""
    tmp = Path(tempfile.mkdtemp(prefix="goldfish-demo-")) / "demo.db"
    print(f"# goldfish demo (throwaway store: {tmp})\n")
    with MemoryEngine(db_path=tmp) as engine:
        for kind, text in DEMO_NOTES:
            m = engine.capture(text, kind=kind, session_id="demo")
            print(f'$ goldfish add --kind {kind} "..." -> saved memory #{m.id}')
        print('\n$ goldfish search "which storage did we pick and why"')
        for r in engine.search("which storage did we pick and why", k=3):
            print(f"  [{r.score:+.3f}] ({r.memory.kind}) {r.memory.content}")
        print('\n$ goldfish inject "start work on the offline test suite"')
        print(engine.inject("start work on the offline test suite", k=3).text)
        stats = engine.stats()
        print(f"\n$ goldfish stats\n  active memories: {stats['active']}")
        print(f"  embedder: {stats['embedder']}")
    print("\nDemo store discarded. Run `goldfish init` in a project to start remembering.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``goldfish`` CLI."""
    parser = argparse.ArgumentParser(
        prog="goldfish",
        description="On-device persistent memory for AI coding agents. "
        "Everything stays on this machine.",
    )
    parser.add_argument("--db", help="database path (default .goldfish/memory.db, or $GOLDFISH_DB)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create a memory store in this project")
    p.set_defaults(func=_cmd_init)

    p = sub.add_parser("add", help="capture a memory (or pipe text on stdin)")
    p.add_argument("text", nargs="?")
    p.add_argument("--kind", default="note", help="category, e.g. note/decision/error")
    p.add_argument("--session", help="agent session id")
    p.add_argument("--pin", action="store_true", help="protect from auto-compaction")
    p.set_defaults(func=_cmd_add)

    p = sub.add_parser("search", help="semantic search over memories")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--kind")
    p.add_argument("--session")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_search)

    p = sub.add_parser("inject", help="print the memory block for the agent's next prompt")
    p.add_argument("task", nargs="?")
    p.add_argument("-k", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=800)
    p.set_defaults(func=_cmd_inject)

    p = sub.add_parser("compact", help="summarize and archive old memories now")
    p.add_argument("--older-than", type=float, metavar="DAYS")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_compact)

    p = sub.add_parser("recent", help="list the newest memories")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--all", action="store_true", help="include archived memories")
    p.set_defaults(func=_cmd_recent)

    p = sub.add_parser("pin", help="protect a memory from compaction")
    p.add_argument("id", type=int)
    p.add_argument("--unpin", action="store_true")
    p.set_defaults(func=_cmd_pin)

    p = sub.add_parser("forget", help="permanently delete a memory")
    p.add_argument("id", type=int)
    p.set_defaults(func=_cmd_forget)

    p = sub.add_parser("stats", help="show store statistics")
    p.set_defaults(func=_cmd_stats)

    p = sub.add_parser("demo", help="tour the features on a throwaway store")
    p.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ModelNotProvisionedError as exc:
        print(f"model not available: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
