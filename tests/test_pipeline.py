import threading

import pytest

from ragcache import runtime_config
from ragcache.llm import LLMError
from ragcache.pipeline import PipelineError, validate_answer

from .conftest import FakeProvider


def test_miss_then_hit_with_cost_accounting(pipeline, corpus, providers):
    r1 = pipeline.query("kb", "how long does shipping take")
    assert not r1.cache_hit and r1.citations
    # small tier: (1000 * 1.0 + 100 * 2.0) / 1e6
    assert r1.model == "small-m" and r1.cost_usd == pytest.approx(0.0012)
    r2 = pipeline.query("kb", "how long does shipping take")
    assert r2.cache_hit and r2.cost_usd == 0.0
    assert r2.saved_cost_usd == pytest.approx(r1.cost_usd)
    assert len(providers["small"].calls) == 1


def test_request_log_records_both_requests(pipeline, corpus):
    pipeline.query("kb", "how long does shipping take")
    pipeline.query("kb", "how long does shipping take")
    rows = pipeline.db.conn.execute(
        "SELECT cache_hit, cost_usd, saved_cost_usd, query_text FROM requests ORDER BY id"
    ).fetchall()
    assert [r["cache_hit"] for r in rows] == [0, 1]
    assert rows[0]["cost_usd"] > 0 and rows[1]["saved_cost_usd"] > 0
    assert all(r["query_text"] is None for r in rows)  # query text is not logged by default


def test_complex_query_routes_to_large(pipeline, corpus, providers):
    q = (
        "compare the refund policy and the shipping policy, and explain why both matter, "
        "and what happens when a refund and a delayed shipment overlap?"
    )
    r = pipeline.query("kb", q)
    assert r.route_tier == "large" and r.model == "large-m"
    assert len(providers["large"].calls) == 1 and not providers["small"].calls


def test_invalid_answer_escalates_to_large_and_costs_both(pipeline, corpus, providers):
    providers["small"].script = lambda user: "I think it is five days."  # no citation
    r = pipeline.query("kb", "how long does shipping take")
    assert r.escalated and r.model == "large-m"
    expected = (1000 * 1 + 100 * 2) / 1e6 + (1000 * 10 + 100 * 20) / 1e6
    assert r.cost_usd == pytest.approx(expected)


def test_answer_invalid_after_escalation_errors_and_is_not_cached(pipeline, corpus, providers):
    for p in providers.values():
        p.script = lambda user: "no citation at all"
    with pytest.raises(PipelineError) as exc:
        pipeline.query("kb", "how long does shipping take")
    assert exc.value.status == 502
    assert pipeline.db.conn.execute("SELECT COUNT(*) c FROM cache_entries").fetchone()["c"] == 0
    row = pipeline.db.conn.execute("SELECT status, error FROM requests").fetchone()
    assert row["status"] == "error" and "citations" in row["error"]


def test_abstention_is_returned_but_never_cached(pipeline, corpus, providers):
    providers["small"].script = lambda user: "INSUFFICIENT_CONTEXT"
    r = pipeline.query("kb", "what is the capital of mars")
    assert r.abstained and not r.escalated
    assert pipeline.db.conn.execute("SELECT COUNT(*) c FROM cache_entries").fetchone()["c"] == 0


def test_provider_failure_fails_over_to_other_tier(pipeline, corpus, providers):
    def boom(user):
        raise LLMError("upstream 503", retryable=True)

    providers["small"].script = boom
    r = pipeline.query("kb", "how long does shipping take")
    assert r.model == "large-m"
    assert len(providers["small"].calls) == 2  # 1 try + 1 retry (llm_max_retries=1)


def test_all_providers_failing_returns_502_and_logs_error(pipeline, corpus, providers):
    def boom(user):
        raise LLMError("down", retryable=False)

    for p in providers.values():
        p.script = boom
    with pytest.raises(PipelineError) as exc:
        pipeline.query("kb", "how long does shipping take")
    assert exc.value.status == 502
    errors = pipeline.db.conn.execute(
        "SELECT COUNT(*) c FROM requests WHERE status='error'"
    ).fetchone()["c"]
    assert errors == 1


def test_unknown_collection_and_empty_query(pipeline, corpus):
    with pytest.raises(PipelineError) as e1:
        pipeline.query("nope", "hi")
    assert e1.value.status == 404
    with pytest.raises(PipelineError) as e2:
        pipeline.query("kb", "   ")
    assert e2.value.status == 422


def test_empty_collection_abstains_without_calling_llm(pipeline, providers):
    pipeline.db.conn.execute(
        "INSERT INTO collections(name, corpus_version, created_at) VALUES ('empty',1,0)"
    )
    r = pipeline.query("empty", "anything")
    assert r.abstained and not providers["small"].calls and not providers["large"].calls


def test_reingest_invalidates_cached_answers(pipeline, corpus, providers):
    from ragcache.retrieval import ingest_document

    pipeline.query("kb", "how long does shipping take")
    cfg = pipeline.config()
    ingest_document(
        pipeline.db.conn,
        pipeline.embedder,
        collection="kb",
        external_id="shipping",
        title="shipping",
        text="Shipping now takes 3 business days.",
        chunk_chars=cfg.chunk_chars,
        overlap=cfg.chunk_overlap_chars,
    )
    r = pipeline.query("kb", "how long does shipping take")
    assert not r.cache_hit and len(providers["small"].calls) == 2


def test_runtime_config_change_takes_effect_without_restart(pipeline, corpus, providers):
    pipeline.query("kb", "how long does shipping take")
    runtime_config.update(pipeline.db.conn, {"cache_enabled": False})
    r = pipeline.query("kb", "how long does shipping take")
    assert not r.cache_hit and len(providers["small"].calls) == 2


def test_concurrent_identical_queries_cost_one_llm_call(pipeline, corpus, providers):
    slow = FakeProvider("small", delay=0.3)
    large = providers["large"]
    pipeline._provider_factory = lambda spec: slow if spec.name == "small-m" else large
    results: list = []

    def run():
        results.append(pipeline.query("kb", "how long does shipping take"))

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 6
    assert len(slow.calls) == 1
    assert sum(1 for r in results if r.cache_hit) == 5


def test_validate_answer_rules():
    assert validate_answer("Shipping takes five days [1] and [2].", 2, "ABSTAIN") == (
        True,
        False,
        [1, 2],
    )
    assert validate_answer("ABSTAIN", 2, "ABSTAIN") == (True, True, [])
    assert (
        validate_answer("Shipping takes five days [3].", 2, "ABSTAIN")[0] is False
    )  # cites a missing passage
    assert validate_answer("Yes.", 2, "ABSTAIN")[0] is False
    assert validate_answer("Shipping takes five days [0].", 2, "ABSTAIN")[0] is False


def test_validate_answer_is_lenient_about_abstention_and_rejects_empty_answers():
    assert validate_answer("INSUFFICIENT CONTEXT", 2, "INSUFFICIENT_CONTEXT") == (True, True, [])
    assert validate_answer("insufficient_context.", 2, "INSUFFICIENT_CONTEXT")[1] is True
    assert validate_answer("[1]", 2, "INSUFFICIENT_CONTEXT")[0] is False  # citation with no content
    assert validate_answer("[1] ok", 2, "INSUFFICIENT_CONTEXT")[0] is False
    assert validate_answer("[1] Shipping takes five days.", 2, "INSUFFICIENT_CONTEXT")[0] is True


def test_rejected_output_is_visible_in_the_error_for_debugging(pipeline, corpus, providers):
    for p in providers.values():
        p.script = lambda user: "totally uncited rambling"
    with pytest.raises(PipelineError) as exc:
        pipeline.query("kb", "how long does shipping take")
    assert "totally uncited rambling" in str(exc.value)
