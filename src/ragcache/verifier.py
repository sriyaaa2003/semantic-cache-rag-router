"""Second-stage verifier for cache hits.

The embedding similarity picks candidates; a cross-encoder then reads both questions together and
says whether they ask the same thing. Only candidates the verifier accepts are served.

The default model, ``cross-encoder/quora-distilroberta-base``, was trained on Quora duplicate
questions. It needs the optional ``verifier`` extra (``pip install -e ".[verifier]"``).
"""

from __future__ import annotations

import threading
from typing import Any, Protocol


class Verifier(Protocol):
    def score(self, query: str, cached_query: str) -> float:
        """Probability-like score in [0, 1] that the two questions mean the same."""
        ...


class CrossEncoderVerifier:
    def __init__(self, model_name: str = "cross-encoder/quora-distilroberta-base") -> None:
        self.model_name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, cached_query: str) -> float:
        return float(self._load().predict([(query, cached_query)], show_progress_bar=False)[0])

    def score_many(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        scores = self._load().predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
