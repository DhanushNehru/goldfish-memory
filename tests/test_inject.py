"""Injection block formatting and token budget."""


def test_inject_formats_markdown_block(engine):
    engine.capture("auth uses jwt tokens stored in cookies", kind="decision")
    block = engine.inject("add logout to auth", k=3)
    assert block.text.startswith("## Relevant memory")
    assert "jwt tokens" in block.text
    assert block.items and block.token_estimate > 0


def test_inject_respects_token_budget(engine):
    for i in range(10):
        engine.capture(f"long note number {i} about the caching layer and its eviction policy")
    block = engine.inject("caching layer", k=10, max_tokens=60)
    assert block.token_estimate <= 60 + 10  # block fits budget plus header
    assert 0 < len(block.items) < 10


def test_inject_tight_budget_truncates_top_hit(engine):
    engine.capture("a fairly long note about the caching layer and eviction policies")
    block = engine.inject("caching", k=5, max_tokens=30)
    assert block.items, "top hit should be truncated, never hidden"
    assert "no relevant memories" not in block.text


def test_inject_empty_store(engine):
    block = engine.inject("brand new task")
    assert "no relevant memories" in block.text
    assert block.items == []
