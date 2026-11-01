"""Automatic compaction: clustering, summarization, archiving, audit log."""

import time


def add_cluster(engine, base_text, n=5, old=True, pinned_first=False):
    ids = []
    for i in range(n):
        m = engine.capture(f"{base_text} variant {i} with the same core topic words")
        ids.append(m.id)
    if pinned_first:
        engine.pin(ids[0])
    if old:
        engine.db.conn.execute(
            f"UPDATE memories SET created_at = ? WHERE id IN ({','.join('?' * len(ids))})",
            [time.time() - 40 * 86400, *ids],
        )
        engine.db.conn.commit()
    return ids


def test_compact_creates_summary_and_archives_children(engine):
    ids = add_cluster(engine, "the payment service uses stripe webhooks")
    result = engine.compact(min_age_days=14)
    assert result.summaries_created == 1
    assert result.memories_archived == len(ids)
    active = engine.recent(50)
    assert len(active) == 1 and active[0].kind == "summary"
    assert active[0].metadata["compacted"] == ids
    archived = engine.recent(50, include_archived=True)
    assert len(archived) == 1 + len(ids)


def test_compact_dry_run_changes_nothing(engine):
    ids = add_cluster(engine, "the payment service uses stripe webhooks")
    result = engine.compact(min_age_days=14, dry_run=True)
    assert result.summaries_created == 1
    assert engine.db.count() == len(ids)
    assert all(not m.archived for m in engine.recent(50))


def test_compact_skips_pinned_and_recent(engine):
    ids = add_cluster(engine, "kubernetes deployment rollout strategy", pinned_first=True)
    engine.capture("fresh note, not old enough")
    result = engine.compact(min_age_days=14)
    assert result.memories_archived == len(ids) - 1  # pinned one survives
    pinned = engine.db.get(ids[0])
    assert not pinned.archived
    summaries = [m for m in engine.recent(50) if m.kind == "summary"]
    assert ids[0] not in summaries[0].metadata["compacted"]


def test_compact_ignores_small_clusters(engine):
    ids = add_cluster(engine, "lonely topic barely repeated", n=2)
    result = engine.compact(min_age_days=14)
    assert result.summaries_created == 0
    assert engine.db.count() == len(ids)


def test_auto_compact_triggers_on_capture(tmp_path, toy):
    from goldfish_memory import Config, MemoryEngine

    eng = MemoryEngine(
        db_path=tmp_path / "m.db",
        embedder=toy,
        config=Config(auto_compact=True, compact_batch=5, compact_interval_hours=0),
    )
    for i in range(5):
        eng.capture(f"repeated topic about the billing retry loop variant {i}")
    eng.db.conn.execute("UPDATE memories SET created_at = 0")
    eng.db.conn.commit()
    eng.capture("one more capture to trigger the hook")
    summaries = [m for m in eng.recent(50) if m.kind == "summary"]
    assert summaries, "auto-compaction should have run during capture"
    assert eng.db.meta_get("last_compaction") is not None
    eng.close()
