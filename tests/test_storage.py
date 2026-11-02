"""Storage layer: schema, CRUD, archiving, deletion, metadata."""

import numpy as np

from goldfish_memory.db import Database
from goldfish_memory.models import Memory


def make_db(tmp_path):
    return Database(tmp_path / "memory.db")


def vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_add_and_get_roundtrip(tmp_path):
    db = make_db(tmp_path)
    mid = db.add(Memory(content="hello store", kind="note", metadata={"f": 1}), vec(1))
    m = db.get(mid)
    assert m is not None and m.content == "hello store"
    assert m.kind == "note" and m.metadata == {"f": 1}
    assert not m.archived and not m.pinned
    assert m.created_at > 0
    db.close()


def test_recent_orders_newest_first_and_hides_archived(tmp_path):
    db = make_db(tmp_path)
    ids = [db.add(Memory(content=f"m{i}"), vec(i)) for i in range(5)]
    db.set_archived([ids[4]])
    recent = db.recent(10)
    assert [m.content for m in recent] == ["m3", "m2", "m1", "m0"]
    assert len(db.recent(10, include_archived=True)) == 5
    db.close()


def test_vector_search_returns_nearest(tmp_path):
    db = make_db(tmp_path)
    base = vec(7)
    near = base + 0.01 * vec(8)
    near /= np.linalg.norm(near)
    far = vec(9)
    for i, v in enumerate([base, near, far]):
        db.add(Memory(content=f"doc{i}"), v)
    hits = db.search_vec(base, k=2)
    assert [m.content for m, _ in hits] == ["doc0", "doc1"]
    assert hits[0][1] > hits[1][1] > 0.9  # cosine similarity scores
    db.close()


def test_delete_removes_memory_and_vector(tmp_path):
    db = make_db(tmp_path)
    mid = db.add(Memory(content="gone"), vec(3))
    assert db.delete(mid) is True
    assert db.get(mid) is None
    assert db.search_vec(vec(3), k=5) == []
    assert db.delete(9999) is False
    db.close()


def test_pin_and_meta(tmp_path):
    db = make_db(tmp_path)
    mid = db.add(Memory(content="pin me"), vec(4))
    db.set_pinned(mid, True)
    assert db.get(mid).pinned is True
    db.set_pinned(mid, False)
    assert db.get(mid).pinned is False
    assert db.meta_get("nope") is None
    db.meta_set("last_compaction", "123.0")
    assert db.meta_get("last_compaction") == "123.0"
    db.close()


def test_unarchived_older_than_respects_flags(tmp_path):
    db = make_db(tmp_path)
    old = db.add(Memory(content="old"), vec(5))
    pinned = db.add(Memory(content="old but pinned"), vec(6))
    db.add(Memory(content="summary", kind="summary"), vec(7))
    db.set_pinned(pinned, True)
    db.conn.execute("UPDATE memories SET created_at = 0")
    db.conn.commit()
    eligible = db.unarchived_older_than(cutoff=1.0)
    eligible_ids = {m.id for m in eligible}
    assert old in eligible_ids and pinned not in eligible_ids
    assert all(m.kind != "summary" for m in eligible)
    db.close()


def test_count(tmp_path):
    db = make_db(tmp_path)
    ids = [db.add(Memory(content=f"c{i}"), vec(10 + i)) for i in range(3)]
    db.set_archived([ids[0]])
    assert db.count() == 2
    assert db.count(include_archived=True) == 3
    db.close()
