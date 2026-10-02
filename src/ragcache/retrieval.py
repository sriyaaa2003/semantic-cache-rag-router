"""Chunking, ingestion and dense retrieval over a collection."""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .db import Database, transaction
from .embeddings import Embedder, from_blob, to_blob


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    document_external_id: str
    title: str
    text: str
    score: float


def chunk_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """Split on paragraph then sentence boundaries into chunks of about ``chunk_chars``."""
    if overlap >= chunk_chars:
        raise ValueError("overlap must be smaller than chunk size")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        while len(sent) > chunk_chars:  # a single over-long sentence
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:chunk_chars])
            sent = sent[chunk_chars - overlap :]
        if current and len(current) + len(sent) + 1 > chunk_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + sent).strip()
        else:
            current = f"{current} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


def get_or_create_collection(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute(
        "INSERT INTO collections(name, corpus_version, created_at) VALUES (?, 1, ?)",
        (name, time.time()),
    )
    return int(cur.lastrowid or 0)


def ingest_document(
    conn: sqlite3.Connection,
    embedder: Embedder,
    *,
    collection: str,
    external_id: str,
    title: str,
    text: str,
    chunk_chars: int,
    overlap: int,
) -> bool:
    """Idempotently upsert one document. Returns False if the content is unchanged.

    Any change bumps the collection's ``corpus_version`` in the same transaction, which
    invalidates every cached answer for that collection.
    """
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    pieces = chunk_text(text, chunk_chars, overlap)
    if not pieces:
        raise ValueError("document has no content")
    vectors = embedder.embed(pieces)  # outside the transaction: slow, no DB lock held
    with transaction(conn):
        cid = get_or_create_collection(conn, collection)
        existing = conn.execute(
            "SELECT id, content_hash FROM documents WHERE collection_id=? AND external_id=?",
            (cid, external_id),
        ).fetchone()
        if existing and existing["content_hash"] == content_hash:
            return False
        if existing:
            conn.execute("DELETE FROM documents WHERE id=?", (existing["id"],))
        cur = conn.execute(
            "INSERT INTO documents(collection_id, external_id, title, content_hash, created_at) "
            "VALUES (?,?,?,?,?)",
            (cid, external_id, title, content_hash, time.time()),
        )
        doc_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO chunks(document_id, collection_id, ordinal, text, embedding) "
            "VALUES (?,?,?,?,?)",
            [
                (doc_id, cid, i, t, to_blob(v))
                for i, (t, v) in enumerate(zip(pieces, vectors, strict=True))
            ],
        )
        conn.execute(
            "UPDATE collections SET corpus_version = corpus_version + 1 WHERE id=?", (cid,)
        )
    return True


class Retriever:
    """Exact cosine search with an in-memory matrix per (collection, corpus_version)."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._mats: dict[int, tuple[int, NDArray[np.float32], list[sqlite3.Row]]] = {}

    def _load(
        self, collection_id: int, version: int
    ) -> tuple[NDArray[np.float32], list[sqlite3.Row]]:
        with self._lock:
            cached = self._mats.get(collection_id)
            if cached and cached[0] == version:
                return cached[1], cached[2]
        rows = self._db.conn.execute(
            "SELECT c.id, c.text, c.embedding, d.external_id, d.title FROM chunks c "
            "JOIN documents d ON d.id = c.document_id WHERE c.collection_id=? ORDER BY c.id",
            (collection_id,),
        ).fetchall()
        mat = (
            np.stack([from_blob(r["embedding"]) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        with self._lock:
            self._mats[collection_id] = (version, mat, rows)
        return mat, rows

    def search(
        self, collection_id: int, version: int, query_vec: NDArray[np.float32], k: int
    ) -> list[RetrievedChunk]:
        mat, rows = self._load(collection_id, version)
        if len(rows) == 0:
            return []
        scores = mat @ query_vec
        k = min(k, len(rows))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            RetrievedChunk(
                chunk_id=int(rows[i]["id"]),
                document_external_id=rows[i]["external_id"],
                title=rows[i]["title"],
                text=rows[i]["text"],
                score=float(scores[i]),
            )
            for i in top
        ]
