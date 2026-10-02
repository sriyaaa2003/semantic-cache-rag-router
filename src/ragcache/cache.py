"""Semantic answer cache.

A cached answer is reused only when all of these hold:
  1. it is in the same namespace: (collection, corpus_version), so re-ingesting invalidates it;
  2. it has not expired;
  3. the cosine similarity of the query embeddings is >= the configured threshold;
  4. (guard) the queries agree on numbers and negation terms. Embedding models place
     "refund within 30 days" and "refund within 60 days" very close together; the guard stops
     these from being served each other's answers.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .db import Database
from .embeddings import from_blob, to_blob

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z']+")


def query_signature(query: str, negation_terms: list[str]) -> str:
    q = query.lower()
    numbers = sorted(set(_NUMBER.findall(q)))
    negs = sorted(set(_WORD.findall(q)) & {t.lower() for t in negation_terms})
    return json.dumps({"n": numbers, "neg": negs}, sort_keys=True)


@dataclass(frozen=True)
class CachedAnswer:
    entry_id: int
    answer: str
    citations: list[dict[str, object]]
    source_model: str
    original_cost_usd: float
    original_latency_ms: float


@dataclass(frozen=True)
class LookupResult:
    hit: CachedAnswer | None
    best_similarity: float | None
    guard_rejected: bool = False


@dataclass
class _Namespace:
    mat: NDArray[np.float32]
    ids: list[int]
    signatures: list[str]
    expires: list[float]


class SemanticCache:
    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._ns: dict[tuple[int, int], _Namespace] = {}

    def _namespace(self, collection_id: int, version: int) -> _Namespace:
        key = (collection_id, version)
        ns = self._ns.get(key)
        if ns is not None:
            return ns
        rows = self._db.conn.execute(
            "SELECT id, embedding, query_signature, expires_at FROM cache_entries "
            "WHERE collection_id=? AND corpus_version=? ORDER BY id",
            (collection_id, version),
        ).fetchall()
        mat = (
            np.stack([from_blob(r["embedding"]) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        ns = _Namespace(
            mat,
            [r["id"] for r in rows],
            [r["query_signature"] for r in rows],
            [r["expires_at"] for r in rows],
        )
        self._ns[key] = ns
        return ns

    def lookup(
        self,
        collection_id: int,
        version: int,
        query_vec: NDArray[np.float32],
        signature: str,
        *,
        threshold: float,
        guard: bool,
        now: float | None = None,
    ) -> LookupResult:
        now = time.time() if now is None else now
        with self._lock:
            ns = self._namespace(collection_id, version)
            if not ns.ids:
                return LookupResult(None, None)
            sims = ns.mat @ query_vec
            expires = np.asarray(ns.expires)
            sims = np.where(expires > now, sims, -1.0)  # expired entries can never match
            order = np.argsort(-sims)
            best = float(sims[order[0]])
            guard_rejected = False
            chosen: tuple[int, float] | None = None
            for idx in order:
                sim = float(sims[idx])
                if sim < threshold:
                    break
                if guard and ns.signatures[idx] != signature:
                    guard_rejected = True
                    continue
                chosen = (ns.ids[idx], sim)
                break
        if chosen is None:
            return LookupResult(None, best, guard_rejected)
        entry_id, sim = chosen
        row = self._db.conn.execute(
            "SELECT * FROM cache_entries WHERE id=?", (entry_id,)
        ).fetchone()
        if row is None:  # deleted by another worker since the matrix was built
            return LookupResult(None, best, guard_rejected)
        self._db.conn.execute(
            "UPDATE cache_entries SET hit_count = hit_count + 1, last_hit_at=? WHERE id=?",
            (now, entry_id),
        )
        return LookupResult(
            CachedAnswer(
                entry_id=entry_id,
                answer=row["answer"],
                citations=json.loads(row["citations_json"]),
                source_model=row["source_model"],
                original_cost_usd=row["original_cost_usd"],
                original_latency_ms=row["original_latency_ms"],
            ),
            sim,
            guard_rejected,
        )

    def store(
        self,
        collection_id: int,
        version: int,
        query_text: str,
        query_vec: NDArray[np.float32],
        signature: str,
        *,
        answer: str,
        citations: list[dict[str, object]],
        source_model: str,
        cost_usd: float,
        latency_ms: float,
        ttl_seconds: int,
        max_entries: int,
        now: float | None = None,
    ) -> int:
        now = time.time() if now is None else now
        expires_at = now + ttl_seconds
        cur = self._db.conn.execute(
            "INSERT INTO cache_entries(collection_id, corpus_version, query_text, query_signature, "
            "embedding, answer, citations_json, source_model, original_cost_usd, "
            "original_latency_ms, created_at, expires_at, last_hit_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                collection_id,
                version,
                query_text,
                signature,
                to_blob(query_vec),
                answer,
                json.dumps(citations),
                source_model,
                cost_usd,
                latency_ms,
                now,
                expires_at,
                now,
            ),
        )
        entry_id = int(cur.lastrowid or 0)
        with self._lock:
            ns = self._namespace(collection_id, version)
            if entry_id not in ns.ids:  # _namespace may already have loaded the new row
                ns.mat = (
                    np.vstack([ns.mat, query_vec[None, :]]) if ns.ids else query_vec[None, :].copy()
                )
                ns.ids.append(entry_id)
                ns.signatures.append(signature)
                ns.expires.append(expires_at)
            if len(ns.ids) > max_entries:
                self._evict(collection_id, version, ns, len(ns.ids) - max_entries)
        return entry_id

    def _evict(self, collection_id: int, version: int, ns: _Namespace, n: int) -> None:
        """Drop the ``n`` least-recently-hit entries. Caller holds the lock."""
        rows = self._db.conn.execute(
            "SELECT id FROM cache_entries WHERE collection_id=? AND corpus_version=? "
            "ORDER BY last_hit_at ASC, id ASC LIMIT ?",
            (collection_id, version, n),
        ).fetchall()
        victims = {r["id"] for r in rows}
        self._db.conn.executemany("DELETE FROM cache_entries WHERE id=?", [(v,) for v in victims])
        keep = [i for i, eid in enumerate(ns.ids) if eid not in victims]
        ns.mat = ns.mat[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        ns.ids = [ns.ids[i] for i in keep]
        ns.signatures = [ns.signatures[i] for i in keep]
        ns.expires = [ns.expires[i] for i in keep]

    def purge_stale(self, current_versions: dict[int, int], now: float | None = None) -> int:
        """Delete expired entries and entries from superseded corpus versions."""
        now = time.time() if now is None else now
        cur = self._db.conn.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (now,))
        removed = cur.rowcount
        for cid, version in current_versions.items():
            cur = self._db.conn.execute(
                "DELETE FROM cache_entries WHERE collection_id=? AND corpus_version<?",
                (cid, version),
            )
            removed += cur.rowcount
        with self._lock:
            self._ns.clear()
        return removed
