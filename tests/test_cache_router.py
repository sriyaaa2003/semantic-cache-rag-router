import numpy as np

from ragcache.cache import SemanticCache, query_signature
from ragcache.router import decide
from ragcache.runtime_config import RuntimeConfig

from .conftest import HashEmbedder

NEG = RuntimeConfig().negation_terms


def _store(cache, emb, text, answer="a", version=1, now=1000.0, ttl=100, max_entries=100):
    v = emb.embed([text])[0]
    return cache.store(
        1,
        version,
        text,
        v,
        query_signature(text, NEG),
        answer=answer,
        citations=[],
        source_model="m",
        cost_usd=0.5,
        latency_ms=100.0,
        ttl_seconds=ttl,
        max_entries=max_entries,
        now=now,
    )


def _lookup(cache, emb, text, version=1, now=1001.0, threshold=0.8, guard=True):
    v = emb.embed([text])[0]
    return cache.lookup(
        1, version, v, query_signature(text, NEG), threshold=threshold, guard=guard, now=now
    )


def _cache(db):
    db.conn.execute(
        "INSERT INTO collections(id, name, corpus_version, created_at) VALUES (1,'c',1,0)"
    )
    return SemanticCache(db)


def test_exact_and_paraphrase_hit_unrelated_miss(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "how do i reset my password")
    assert _lookup(cache, emb, "how do i reset my password").hit is not None
    assert _lookup(cache, emb, "reset my password how do i").hit is not None
    assert _lookup(cache, emb, "what is the weather in rome today").hit is None


def test_guard_blocks_different_numbers_and_negation(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "can i get a refund within 30 days")
    near_number = "can i get a refund within 60 days"
    r = _lookup(cache, emb, near_number, threshold=0.5, guard=True)
    assert r.hit is None and r.guard_rejected
    assert _lookup(cache, emb, near_number, threshold=0.5, guard=False).hit is not None

    _store(cache, emb, "is the warranty valid abroad")
    r = _lookup(cache, emb, "is the warranty not valid abroad", threshold=0.5, guard=True)
    assert r.hit is None and r.guard_rejected


def test_expired_entries_never_match(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "hello there friend", now=1000.0, ttl=10)
    assert _lookup(cache, emb, "hello there friend", now=1005.0).hit is not None
    assert _lookup(cache, emb, "hello there friend", now=1011.0).hit is None


def test_new_corpus_version_is_a_different_namespace(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "shipping time", version=1)
    assert _lookup(cache, emb, "shipping time", version=2).hit is None


def test_lru_eviction_keeps_recently_hit_entries(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "alpha alpha alpha", now=1000.0, max_entries=2)
    _store(cache, emb, "bravo bravo bravo", now=1001.0, max_entries=2)
    assert _lookup(cache, emb, "alpha alpha alpha", now=1002.0).hit is not None  # refresh alpha
    _store(cache, emb, "charlie charlie charlie", now=1003.0, max_entries=2)
    assert _lookup(cache, emb, "bravo bravo bravo", now=1004.0).hit is None  # evicted
    assert _lookup(cache, emb, "alpha alpha alpha", now=1004.0).hit is not None
    assert db.conn.execute("SELECT COUNT(*) c FROM cache_entries").fetchone()["c"] == 2


def test_cache_state_survives_restart(db):
    emb = HashEmbedder()
    _store(_cache(db), emb, "persisted question")
    fresh = SemanticCache(db)  # new process: the matrix is rebuilt from the database
    assert _lookup(fresh, emb, "persisted question").hit is not None


def test_signature_ignores_case_and_irrelevant_word_order():
    assert query_signature("Refund in 30 DAYS?", NEG) == query_signature("refund in 30 days", NEG)
    assert query_signature("in 30 days refund", NEG) == query_signature("refund in 30 days", NEG)
    assert query_signature("refund in 30 days", NEG) != query_signature("refund in 31 days", NEG)


def test_router_simple_vs_complex():
    cfg = RuntimeConfig()
    simple = decide("how long is shipping", 0.9, cfg)
    assert simple.tier == "small"
    hard = decide(
        "compare the refund policy and the warranty policy, and explain why they differ, "
        "and what happens when both apply to one order?",
        0.45,
        cfg,
    )
    assert hard.tier == "large" and hard.reasons


def test_router_no_context_pushes_toward_large():
    cfg = RuntimeConfig(router_large_threshold=0.2)
    d = decide("anything", None, cfg)
    assert d.tier == "large"
    expected = cfg.router_weights.low_retrieval_confidence + cfg.router_weights.length / (
        cfg.router_length_reference_tokens
    )
    assert np.isclose(d.score, expected, atol=1e-3)
