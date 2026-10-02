"""HTTP API. Route handlers only validate, authenticate and delegate; logic lives in the pipeline."""

from __future__ import annotations

import logging
import secrets
from importlib import resources
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .. import metrics, models, runtime_config
from ..config import Settings, get_settings
from ..db import Database
from ..embeddings import Embedder, FastEmbedEmbedder
from ..pipeline import Pipeline, PipelineError
from ..retrieval import ingest_document

log = logging.getLogger("ragcache")


class QueryRequest(BaseModel):
    collection: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=4_000)


class DocumentRequest(BaseModel):
    external_id: str = Field(min_length=1, max_length=500)
    title: str = Field(default="", max_length=500)
    text: str = Field(min_length=1, max_length=2_000_000)


class ModelRequest(BaseModel):
    name: str = Field(min_length=1)
    provider: str = Field(pattern="^(anthropic|openai_compat)$")
    model_id: str = Field(min_length=1)
    tier: str = Field(pattern="^(small|large)$")
    input_price_per_mtok: float = Field(ge=0)
    output_price_per_mtok: float = Field(ge=0)
    base_url: str | None = None
    api_key_env: str | None = None
    priority: int = 100


def create_app(
    settings: Settings | None = None,
    *,
    pipeline: Pipeline | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    if pipeline is None:
        db = Database(settings.db_path)
        pipeline = Pipeline(db, embedder or FastEmbedEmbedder(settings.embedding_model), settings)
    pipe = pipeline
    app = FastAPI(title="ragcache", version="0.1.0")

    def _check(provided: str | None, expected: Any, what: str) -> None:
        if expected is None:
            raise HTTPException(503, f"{what} API key is not configured on the server")
        if not provided or not secrets.compare_digest(provided, expected.get_secret_value()):
            raise HTTPException(401, "invalid or missing API key")

    def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
        _check(x_api_key, settings.admin_api_key, "admin")

    def require_query(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if (
            settings.query_api_key is None
        ):  # an open query endpoint is an explicit deployment choice
            return
        _check(x_api_key, settings.query_api_key, "query")

    admin = Depends(require_admin)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        try:
            pipe.db.conn.execute("SELECT 1").fetchone()
        except Exception as exc:
            raise HTTPException(503, "database unavailable") from exc
        return {"status": "ready"}

    @app.post("/v1/query", dependencies=[Depends(require_query)])
    def query(body: QueryRequest) -> dict[str, Any]:
        try:
            r = pipe.query(body.collection, body.query)
        except PipelineError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        return {
            "request_id": r.request_id,
            "answer": r.answer,
            "abstained": r.abstained,
            "citations": r.citations,
            "cache": {"hit": r.cache_hit, "similarity": r.similarity},
            "routing": {
                "tier": r.route_tier,
                "model": r.model,
                "escalated": r.escalated,
                "reasons": r.route_reasons,
            },
            "usage": {
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "saved_cost_usd": r.saved_cost_usd,
            },
            "latency_ms": r.latency_ms,
        }

    @app.post("/v1/collections/{name}/documents", dependencies=[admin])
    def add_document(name: str, body: DocumentRequest) -> dict[str, Any]:
        cfg = pipe.config()
        try:
            changed = ingest_document(
                pipe.db.conn,
                pipe.embedder,
                collection=name,
                external_id=body.external_id,
                title=body.title,
                text=body.text,
                chunk_chars=cfg.chunk_chars,
                overlap=cfg.chunk_overlap_chars,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"collection": name, "external_id": body.external_id, "changed": changed}

    @app.get("/v1/admin/config", dependencies=[admin])
    def get_config() -> dict[str, Any]:
        return runtime_config.load(pipe.db.conn).model_dump()

    @app.put("/v1/admin/config", dependencies=[admin])
    def put_config(changes: dict[str, Any]) -> dict[str, Any]:
        try:
            cfg = runtime_config.update(pipe.db.conn, changes)
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            raise HTTPException(422, str(exc)) from exc
        pipe.invalidate_config()
        return cfg.model_dump()

    @app.get("/v1/admin/models", dependencies=[admin])
    def list_models() -> list[dict[str, Any]]:
        return [m.__dict__ for m in models.list_models(pipe.db.conn)]

    @app.post("/v1/admin/models", dependencies=[admin])
    def upsert_model(body: ModelRequest) -> dict[str, str]:
        models.add_model(pipe.db.conn, **body.model_dump())
        return {"status": "ok", "name": body.name}

    @app.get("/v1/metrics/summary", dependencies=[admin])
    def metrics_summary(
        window_seconds: Annotated[float | None, Query(gt=0)] = None,
    ) -> dict[str, object]:
        return metrics.summary(pipe.db.conn, window_seconds)

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return resources.files("ragcache.api").joinpath("dashboard.html").read_text("utf-8")

    return app
