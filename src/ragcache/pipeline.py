"""The request pipeline: cache lookup -> retrieval -> routing -> generation -> validation -> cache.

Failure policy
  * embedding failure           -> request fails (no answer is fabricated); logged with status=error
  * retrieval returns nothing   -> abstain without calling an LLM (no context, no grounded answer)
  * provider failure            -> retry (retryable errors), then fail over to the next model in the
                                   tier, then to the other tier; if all fail the request errors
  * invalid answer (no valid citation, not an abstention) -> escalate once to the large tier
  * an abstention or invalid answer is never cached
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import models as model_registry
from . import runtime_config
from .cache import LookupResult, SemanticCache, query_signature
from .config import Settings
from .db import Database
from .embeddings import Embedder
from .llm import LLMError, LLMProvider, LLMResult, build_provider
from .models import ModelSpec
from .retrieval import RetrievedChunk, Retriever
from .router import RouteDecision, decide

log = logging.getLogger("ragcache")

ProviderFactory = Callable[[ModelSpec], LLMProvider]


class PipelineError(Exception):
    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class QueryResult:
    request_id: str
    answer: str
    abstained: bool
    cache_hit: bool
    similarity: float | None
    route_tier: str | None
    model: str | None
    escalated: bool
    citations: list[dict[str, object]]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    saved_cost_usd: float
    latency_ms: float
    route_reasons: list[str] = field(default_factory=list)


def _context_block(
    chunks: list[RetrievedChunk], max_chars: int
) -> tuple[str, list[RetrievedChunk]]:
    used: list[RetrievedChunk] = []
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        piece = f"[{len(used) + 1}] ({chunk.title or chunk.document_external_id}) {chunk.text}"
        if total + len(piece) > max_chars and used:
            break
        parts.append(piece)
        used.append(chunk)
        total += len(piece)
    return "\n\n".join(parts), used


_CITATION = re.compile(r"\[(\d+)\]")


def validate_answer(text: str, n_passages: int, abstain_token: str) -> tuple[bool, bool, list[int]]:
    """Returns (valid, abstained, cited_passage_numbers). Deterministic, no LLM involved."""
    stripped = text.strip()
    if stripped == abstain_token:
        return True, True, []
    cited = sorted({int(m) for m in _CITATION.findall(stripped)})
    if not cited or any(c < 1 or c > n_passages for c in cited):
        return False, False, cited
    return True, False, cited


class _SingleFlight:
    """Coalesce concurrent identical misses so N identical queries cost one LLM call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}

    def acquire(self, key: str) -> tuple[bool, threading.Event]:
        with self._lock:
            ev = self._inflight.get(key)
            if ev is not None:
                return False, ev
            ev = threading.Event()
            self._inflight[key] = ev
            return True, ev

    def release(self, key: str) -> None:
        with self._lock:
            ev = self._inflight.pop(key, None)
        if ev is not None:
            ev.set()


class Pipeline:
    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        settings: Settings,
        *,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.settings = settings
        self.cache = SemanticCache(db)
        self.retriever = Retriever(db)
        self._flight = _SingleFlight()
        self._provider_factory = provider_factory or self._default_factory
        self._cfg: runtime_config.RuntimeConfig | None = None
        self._cfg_at = 0.0
        self._cfg_lock = threading.Lock()

    # ----- configuration -------------------------------------------------------------------
    def config(self) -> runtime_config.RuntimeConfig:
        with self._cfg_lock:
            now = time.monotonic()
            if self._cfg is None or now - self._cfg_at > self.settings.config_refresh_seconds:
                self._cfg = runtime_config.load(self.db.conn)
                self._cfg_at = now
            return self._cfg

    def invalidate_config(self) -> None:
        with self._cfg_lock:
            self._cfg = None

    def _default_factory(self, spec: ModelSpec) -> LLMProvider:
        return build_provider(
            spec.provider,
            spec.model_id,
            base_url=spec.base_url,
            api_key_env=spec.api_key_env,
            timeout=self.settings.llm_timeout_seconds,
        )

    # ----- generation with retry and failover --------------------------------------------------
    def _generate(
        self, tier: str, system: str, user: str, cfg: runtime_config.RuntimeConfig
    ) -> tuple[ModelSpec, LLMResult]:
        other = "large" if tier == "small" else "small"
        candidates = model_registry.list_models(
            self.db.conn, tier=tier
        ) + model_registry.list_models(self.db.conn, tier=other)
        if not candidates:
            raise PipelineError("no active models configured", status=503)
        last: Exception | None = None
        for spec in candidates:
            for attempt in range(self.settings.llm_max_retries + 1):
                try:
                    provider = self._provider_factory(spec)
                    result = provider.generate(
                        system, user, max_tokens=cfg.max_output_tokens, temperature=cfg.temperature
                    )
                    return spec, result
                except LLMError as exc:
                    last = exc
                    log.warning(
                        "llm_error model=%s attempt=%d retryable=%s err=%s",
                        spec.name,
                        attempt,
                        exc.retryable,
                        exc,
                    )
                    if not exc.retryable:
                        break
                    time.sleep(min(0.5 * 2**attempt, 4.0))
        raise PipelineError(f"all models failed: {last}", status=502)

    # ----- main entry point ------------------------------------------------------------------------
    def query(self, collection: str, query: str) -> QueryResult:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        cfg = self.config()
        row = self.db.conn.execute(
            "SELECT id, corpus_version FROM collections WHERE name=?", (collection,)
        ).fetchone()
        if row is None:
            raise PipelineError(f"unknown collection {collection!r}", status=404)
        cid, version = int(row["id"]), int(row["corpus_version"])
        query = query.strip()
        if not query:
            raise PipelineError("empty query", status=422)

        try:
            result = self._run(request_id, cid, version, query, cfg, started)
        except PipelineError as exc:
            self._log_request(
                request_id,
                cid,
                query,
                cfg,
                status="error",
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        except Exception as exc:
            log.exception("unexpected failure request_id=%s", request_id)
            self._log_request(
                request_id,
                cid,
                query,
                cfg,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise PipelineError("internal error", status=500) from exc
        return result

    def _run(
        self,
        request_id: str,
        cid: int,
        version: int,
        query: str,
        cfg: runtime_config.RuntimeConfig,
        started: float,
    ) -> QueryResult:
        qvec = self.embedder.embed([query])[0]
        sig = query_signature(query, cfg.negation_terms)

        flight_key = f"{cid}:{version}:{hashlib.sha256(query.lower().encode()).hexdigest()}"
        leader = True
        if cfg.cache_enabled:
            lookup = self.cache.lookup(
                cid,
                version,
                qvec,
                sig,
                threshold=cfg.cache_similarity_threshold,
                guard=cfg.cache_guard_enabled,
            )
            if lookup.hit is None:
                leader, event = self._flight.acquire(flight_key)
                if not leader:  # an identical query is already being answered: wait, then re-check
                    event.wait(timeout=self.settings.llm_timeout_seconds * 2)
                    lookup = self.cache.lookup(
                        cid,
                        version,
                        qvec,
                        sig,
                        threshold=cfg.cache_similarity_threshold,
                        guard=cfg.cache_guard_enabled,
                    )
                    leader = lookup.hit is None
            if lookup.hit is not None:
                hit = lookup.hit
                latency = (time.perf_counter() - started) * 1000
                result = QueryResult(
                    request_id=request_id,
                    answer=hit.answer,
                    abstained=False,
                    cache_hit=True,
                    similarity=lookup.best_similarity,
                    route_tier=None,
                    model=hit.source_model,
                    escalated=False,
                    citations=hit.citations,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    saved_cost_usd=hit.original_cost_usd,
                    latency_ms=latency,
                )
                self._log_request(
                    request_id,
                    cid,
                    query,
                    cfg,
                    status="ok",
                    cache_hit=True,
                    similarity=lookup.best_similarity,
                    saved=hit.original_cost_usd,
                    latency_ms=latency,
                    model_name=hit.source_model,
                )
                return result
        else:
            lookup = None

        try:
            return self._answer(request_id, cid, version, query, qvec, sig, cfg, started, lookup)
        finally:
            if cfg.cache_enabled and leader:
                self._flight.release(flight_key)

    def _answer(
        self,
        request_id: str,
        cid: int,
        version: int,
        query: str,
        qvec: np.ndarray,
        sig: str,
        cfg: runtime_config.RuntimeConfig,
        started: float,
        lookup: LookupResult | None,
    ) -> QueryResult:
        t0 = time.perf_counter()
        chunks = self.retriever.search(cid, version, qvec, cfg.top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        similarity = lookup.best_similarity if lookup else None

        if not chunks:
            latency = (time.perf_counter() - started) * 1000
            self._log_request(
                request_id,
                cid,
                query,
                cfg,
                status="ok",
                similarity=similarity,
                latency_ms=latency,
                retrieval_ms=retrieval_ms,
            )
            return QueryResult(
                request_id=request_id,
                answer=cfg.abstain_token,
                abstained=True,
                cache_hit=False,
                similarity=similarity,
                route_tier=None,
                model=None,
                escalated=False,
                citations=[],
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                saved_cost_usd=0.0,
                latency_ms=latency,
                route_reasons=["collection has no content"],
            )

        context, used = _context_block(chunks, cfg.max_context_chars)
        decision: RouteDecision = decide(query, chunks[0].score, cfg)
        system = cfg.system_prompt.format(abstain_token=cfg.abstain_token)
        user = f"Context passages:\n{context}\n\nQuestion: {query}"

        in_tok = out_tok = 0
        cost = 0.0
        gen_ms = 0.0
        escalated = False
        tier = decision.tier
        spec, res = self._generate(tier, system, user, cfg)
        in_tok += res.input_tokens
        out_tok += res.output_tokens
        cost += spec.cost_usd(res.input_tokens, res.output_tokens)
        gen_ms += res.latency_ms
        valid, abstained, cited = validate_answer(res.text, len(used), cfg.abstain_token)

        if not valid and cfg.escalate_on_invalid_answer and spec.tier != "large":
            escalated = True
            tier = "large"
            spec, res = self._generate("large", system, user, cfg)
            in_tok += res.input_tokens
            out_tok += res.output_tokens
            cost += spec.cost_usd(res.input_tokens, res.output_tokens)
            gen_ms += res.latency_ms
            valid, abstained, cited = validate_answer(res.text, len(used), cfg.abstain_token)

        if not valid:
            # Never return or cache an answer that failed validation.
            raise PipelineError("model produced an answer without valid citations", status=502)

        citations = [
            {
                "n": n,
                "chunk_id": used[n - 1].chunk_id,
                "document": used[n - 1].document_external_id,
                "title": used[n - 1].title,
            }
            for n in cited
        ]
        latency = (time.perf_counter() - started) * 1000
        if cfg.cache_enabled and not abstained:
            self.cache.store(
                cid,
                version,
                query,
                qvec,
                sig,
                answer=res.text,
                citations=citations,
                source_model=spec.name,
                cost_usd=cost,
                latency_ms=latency,
                ttl_seconds=cfg.cache_ttl_seconds,
                max_entries=cfg.cache_max_entries_per_namespace,
            )
        self._log_request(
            request_id,
            cid,
            query,
            cfg,
            status="ok",
            similarity=similarity,
            route_tier=tier,
            model_name=spec.name,
            escalated=escalated,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=cost,
            latency_ms=latency,
            retrieval_ms=retrieval_ms,
            generation_ms=gen_ms,
        )
        return QueryResult(
            request_id=request_id,
            answer=res.text,
            abstained=abstained,
            cache_hit=False,
            similarity=similarity,
            route_tier=tier,
            model=spec.name,
            escalated=escalated,
            citations=citations,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            saved_cost_usd=0.0,
            latency_ms=latency,
            route_reasons=decision.reasons,
        )

    # ----- request log ---------------------------------------------------------------------------
    def _log_request(
        self,
        request_id: str,
        cid: int,
        query: str,
        cfg: runtime_config.RuntimeConfig,
        *,
        status: str,
        latency_ms: float,
        error: str | None = None,
        cache_hit: bool = False,
        similarity: float | None = None,
        route_tier: str | None = None,
        model_name: str | None = None,
        escalated: bool = False,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        saved: float = 0.0,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
    ) -> None:
        try:
            self.db.conn.execute(
                "INSERT INTO requests(request_id, ts, collection_id, query_hash, query_text, status, "
                "cache_hit, similarity, route_tier, model_name, escalated, input_tokens, "
                "output_tokens, cost_usd, saved_cost_usd, latency_ms, retrieval_ms, generation_ms, "
                "error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    time.time(),
                    cid,
                    hashlib.sha256(query.encode()).hexdigest(),
                    query if cfg.log_query_text else None,
                    status,
                    int(cache_hit),
                    similarity,
                    route_tier,
                    model_name,
                    int(escalated),
                    input_tokens,
                    output_tokens,
                    cost,
                    saved,
                    latency_ms,
                    retrieval_ms,
                    generation_ms,
                    error,
                ),
            )
        except Exception:  # logging must never break serving
            log.exception("failed to write request log request_id=%s", request_id)
