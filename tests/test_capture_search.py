"""Engine capture and semantic search behavior (toy embedder)."""


def test_capture_returns_id_and_persists(engine):
    m = engine.capture("chose sqlite for storage", kind="decision", metadata={"pr": 12})
    assert m.id == 1
    again = engine.db.get(m.id)
    assert again.content == "chose sqlite for storage"
    assert again.metadata == {"pr": 12}


def test_search_ranks_matching_first(engine):
    engine.capture("database storage sqlite vector index", kind="decision")
    engine.capture("frontend css flexbox layout colors", kind="note")
    engine.capture("storage engine benchmark latency numbers", kind="note")
    hits = engine.search("database storage sqlite decision", k=3)
    assert hits[0].memory.kind == "decision"
    assert hits[0].score > hits[-1].score
    assert len(hits) == 3


def test_search_filters_and_excludes_archived(engine):
    a = engine.capture("alpha topic one", kind="decision")
    engine.capture("alpha topic two", kind="error")
    hits = engine.search("alpha topic", k=5, kind="error")
    assert len(hits) == 1 and hits[0].memory.kind == "error"
    engine.db.set_archived([a.id])
    kinds = {h.memory.kind for h in engine.search("alpha topic", k=5)}
    assert "decision" not in kinds
    with_arch = engine.search("alpha topic", k=5, include_archived=True)
    assert "decision" in {h.memory.kind for h in with_arch}


def test_search_since_days(engine):
    engine.capture("recent note about caching")
    old = engine.capture("ancient note about caching")
    engine.db.conn.execute("UPDATE memories SET created_at = 0 WHERE id = ?", (old.id,))
    engine.db.conn.commit()
    hits = engine.search("caching", k=5, since_days=1)
    assert {h.memory.content for h in hits} == {"recent note about caching"}


def test_search_empty_store(engine):
    assert engine.search("anything") == []
