import json

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ragcache.api.app import create_app
from ragcache.eval_threshold import evaluate, load_pairs, recommend

from .conftest import HashEmbedder

ADMIN = {"x-api-key": "admin-secret"}


@pytest.fixture
def client(pipeline, settings):
    return TestClient(create_app(settings, pipeline=pipeline))


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}


def test_admin_endpoints_require_key(client):
    assert client.get("/v1/admin/config").status_code == 401
    assert client.get("/v1/admin/config", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/v1/admin/config", headers=ADMIN).status_code == 200


def test_admin_disabled_when_no_key_configured(pipeline, settings):
    s = settings.model_copy(update={"admin_api_key": None})
    c = TestClient(create_app(s, pipeline=pipeline))
    assert c.get("/v1/admin/config", headers=ADMIN).status_code == 503


def test_ingest_query_metrics_roundtrip(client):
    r = client.post(
        "/v1/collections/kb/documents",
        headers=ADMIN,
        json={
            "external_id": "d1",
            "title": "Shipping",
            "text": "Standard shipping takes 5 business days.",
        },
    )
    assert r.status_code == 200 and r.json()["changed"] is True
    body = {"collection": "kb", "query": "how long does shipping take"}
    first = client.post("/v1/query", json=body).json()
    second = client.post("/v1/query", json=body).json()
    assert first["cache"]["hit"] is False and second["cache"]["hit"] is True
    assert second["usage"]["saved_cost_usd"] == pytest.approx(first["usage"]["cost_usd"])
    m = client.get("/v1/metrics/summary", headers=ADMIN).json()
    assert m["requests"] == 2 and m["cache_hit_rate"] == 0.5
    assert m["by_route"]["cache_hit"]["requests"] == 1


def test_query_validation_and_errors(client):
    assert client.post("/v1/query", json={"collection": "kb", "query": ""}).status_code == 422
    assert client.post("/v1/query", json={"collection": "x", "query": "x"}).status_code == 404


def test_query_key_enforced_when_configured(pipeline, settings):
    s = settings.model_copy(update={"query_api_key": SecretStr("q-secret")})
    c = TestClient(create_app(s, pipeline=pipeline))
    body = {"collection": "kb", "query": "x"}
    assert c.post("/v1/query", json=body).status_code == 401
    assert c.post("/v1/query", json=body, headers={"x-api-key": "q-secret"}).status_code == 404


def test_config_update_validation(client):
    ok = client.put("/v1/admin/config", headers=ADMIN, json={"cache_similarity_threshold": 0.9})
    assert ok.status_code == 200 and ok.json()["cache_similarity_threshold"] == 0.9
    assert client.put("/v1/admin/config", headers=ADMIN, json={"top_k": 0}).status_code == 422
    assert client.put("/v1/admin/config", headers=ADMIN, json={"bogus": 1}).status_code == 422


def test_models_endpoints(client):
    r = client.post(
        "/v1/admin/models",
        headers=ADMIN,
        json={
            "name": "extra",
            "provider": "openai_compat",
            "model_id": "x",
            "tier": "small",
            "input_price_per_mtok": 0.1,
            "output_price_per_mtok": 0.2,
            "priority": 5,
        },
    )
    assert r.status_code == 200
    names = [m["name"] for m in client.get("/v1/admin/models", headers=ADMIN).json()]
    assert "extra" in names
    bad = client.post(
        "/v1/admin/models",
        headers=ADMIN,
        json={
            "name": "x",
            "provider": "nope",
            "model_id": "x",
            "tier": "small",
            "input_price_per_mtok": 0,
            "output_price_per_mtok": 0,
        },
    )
    assert bad.status_code == 422


def test_dashboard_served(client):
    r = client.get("/dashboard")
    assert r.status_code == 200 and "ragcache" in r.text


def test_threshold_evaluation_math(tmp_path):
    pairs = [
        ("how do i reset my password", "reset my password how do i", True),
        ("refund within 30 days", "refund within 60 days", False),
        ("best pizza in rome", "how to learn rust", False),
        ("is it waterproof", "is it not waterproof", False),
    ]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps({"q1": a, "q2": b, "is_duplicate": d}) for a, b, d in pairs))
    rows = evaluate(load_pairs(str(p)), HashEmbedder(), [0.5, 0.9])
    on = {r.threshold: r for r in rows if r.guard}
    off = {r.threshold: r for r in rows if not r.guard}
    assert on[0.5].false_hits == 0  # the guard removes the number and negation near-misses
    assert off[0.5].false_hits >= 1  # without the guard they would be served
    assert on[0.5].recall == 1.0
    best = recommend(rows, 0.99)
    assert best is not None and best.guard
