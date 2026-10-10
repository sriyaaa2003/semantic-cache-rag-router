from ragcache import runtime_config
from ragcache.pipeline import Pipeline

from .conftest import HashEmbedder, FakeProvider  # noqa: F401
from .test_cache_router import NEG, _cache, _lookup, _store
from ragcache.cache import query_signature


class FakeVerifier:
    """Test double: calls two questions the same only if they share the given keyword."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.calls: list[tuple[str, str]] = []

    def score(self, query: str, cached_query: str) -> float:
        self.calls.append((query, cached_query))
        return 1.0 if self.keyword in query and self.keyword in cached_query else 0.0


def _verified_lookup(cache, emb, text, verifier, min_score=0.5, max_verifications=3):
    v = emb.embed([text])[0]
    return cache.lookup(
        1, 1, v, query_signature(text, NEG), threshold=0.5, guard=False, now=1001.0,
        verify=lambda cached: verifier.score(text, cached) >= min_score,
        max_verifications=max_verifications,
    )


def test_verifier_rejects_candidate_and_accepts_good_one(db):
    emb, cache = HashEmbedder(), _cache(db)
    _store(cache, emb, "how do i reset my password")
    ok = _verified_lookup(cache, emb, "how can i reset my password", FakeVerifier("password"))
    assert ok.hit is not None
    bad = _verified_lookup(cache, emb, "how can i reset my password", FakeVerifier("nothere"))
    assert bad.hit is None and bad.guard_rejected


def test_verifier_call_count_is_capped(db):
    emb, cache = HashEmbedder(), _cache(db)
    for i in range(6):
        _store(cache, emb, f"how do i reset my password variant{i}")
    ver = FakeVerifier("nothere")
    r = _verified_lookup(cache, emb, "how do i reset my password", ver, max_verifications=2)
    assert r.hit is None and len(ver.calls) == 2


def test_pipeline_uses_verifier_when_enabled(db, settings, providers, pipeline, corpus):
    runtime_config.update(
        db.conn,
        {"verifier_enabled": True, "verifier_candidate_threshold": 0.5, "verifier_min_score": 0.5},
    )
    pipeline._verifier = FakeVerifier("shipping")
    pipeline.invalidate_config()
    first = pipeline.query("kb", "How long does shipping take?")
    assert not first.cache_hit
    again = pipeline.query("kb", "How many days does shipping take?")
    assert again.cache_hit  # verifier says same topic
    other = pipeline.query("kb", "What is the refund policy for shipping costs?")
    assert pipeline._verifier.calls  # verifier was consulted
    assert isinstance(pipeline, Pipeline)
    assert other is not None
