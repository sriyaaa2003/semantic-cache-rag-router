"""Test doubles live here only. Production code never imports from tests."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from ragcache import models as model_registry
from ragcache.config import Settings
from ragcache.db import Database
from ragcache.embeddings import normalise
from ragcache.llm import LLMError, LLMResult
from ragcache.pipeline import Pipeline


class HashEmbedder:
    """Deterministic bag-of-words embedder: shared words => similar vectors. Test-only."""

    dim = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9']+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)  # noqa: S324
                out[i, h % self.dim] += 1.0
        return normalise(out)


class FakeProvider:
    """Scripted LLM. ``script`` returns the text (or raises LLMError) for each call."""

    def __init__(
        self,
        name: str,
        script: Callable[[str], str] | None = None,
        in_tokens: int = 1000,
        out_tokens: int = 100,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.calls: list[str] = []
        self.script = script or (lambda user: "The answer is in the passage [1].")
        self.in_tokens, self.out_tokens, self.delay = in_tokens, out_tokens, delay
        self._lock = threading.Lock()

    def generate(self, system: str, user: str, *, max_tokens: int, temperature: float) -> LLMResult:
        with self._lock:
            self.calls.append(user)
        if self.delay:
            time.sleep(self.delay)
        text = self.script(user)
        return LLMResult(text, self.in_tokens, self.out_tokens, 5.0 + self.delay * 1000)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        db_path=str(tmp_path / "t.db"),
        config_refresh_seconds=0.0,
        llm_max_retries=1,
        admin_api_key="admin-secret",
    )


@pytest.fixture
def db(settings):
    return Database(settings.db_path)


@pytest.fixture
def providers():
    return {"small": FakeProvider("small"), "large": FakeProvider("large")}


@pytest.fixture
def pipeline(db, settings, providers):
    model_registry.add_model(
        db.conn,
        name="small-m",
        provider="openai_compat",
        model_id="s",
        tier="small",
        input_price_per_mtok=1.0,
        output_price_per_mtok=2.0,
    )
    model_registry.add_model(
        db.conn,
        name="large-m",
        provider="openai_compat",
        model_id="l",
        tier="large",
        input_price_per_mtok=10.0,
        output_price_per_mtok=20.0,
    )
    by_name = {"small-m": providers["small"], "large-m": providers["large"]}
    return Pipeline(db, HashEmbedder(), settings, provider_factory=lambda spec: by_name[spec.name])


@pytest.fixture
def corpus(pipeline):
    from ragcache.retrieval import ingest_document

    cfg = pipeline.config()
    docs = {
        "refunds": "Refunds are available within 30 days of purchase. Contact support to start one.",
        "shipping": "Standard shipping takes 5 business days. Express shipping takes 2 days.",
        "passwords": "To reset your password open Settings and choose Reset password.",
    }
    for k, text in docs.items():
        ingest_document(
            pipeline.db.conn,
            pipeline.embedder,
            collection="kb",
            external_id=k,
            title=k,
            text=text,
            chunk_chars=cfg.chunk_chars,
            overlap=cfg.chunk_overlap_chars,
        )
    return docs


__all__ = ["FakeProvider", "HashEmbedder", "LLMError"]
