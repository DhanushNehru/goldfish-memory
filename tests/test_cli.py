"""CLI commands end to end."""

import json

from goldfish_memory.cli import main


def test_init_add_search_forget(tmp_path, capsys):
    db = str(tmp_path / "memory.db")
    assert main(["--db", db, "init"]) == 0
    capsys.readouterr()
    assert main(["--db", db, "add", "--kind", "decision", "sqlite stores everything"]) == 0
    assert "saved memory #1" in capsys.readouterr().out
    assert main(["--db", db, "search", "what stores data", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits and hits[0]["content"] == "sqlite stores everything"
    assert main(["--db", db, "forget", "1"]) == 0
    assert main(["--db", db, "search", "sqlite"]) == 0
    assert "no matches" in capsys.readouterr().out


def test_inject_and_stats(tmp_path, capsys):
    db = str(tmp_path / "memory.db")
    main(["--db", db, "add", "redis caches session tokens"])
    capsys.readouterr()
    assert main(["--db", db, "inject", "session caching", "-k", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Relevant memory") and "redis" in out
    assert main(["--db", db, "stats"]) == 0
    assert "active: 1" in capsys.readouterr().out


def test_demo_runs_on_throwaway_store(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "goldfish demo" in out and "Relevant memory" in out
