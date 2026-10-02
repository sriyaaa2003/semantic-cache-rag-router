"""Choose the cache similarity threshold from labelled duplicate / non-duplicate query pairs.

A *false hit* (serving a cached answer to a question that is not really the same) is far worse than
a missed hit, so the recommendation maximises recall subject to a precision floor. This only needs
embeddings, not an LLM, so the numbers it reports are measured, not estimated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import query_signature
from .embeddings import Embedder
from .runtime_config import RuntimeConfig


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    guard: bool
    precision: float
    recall: float
    f1: float
    hits: int
    false_hits: int


def load_pairs(path: str) -> list[tuple[str, str, bool]]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            pairs.append((d["q1"], d["q2"], bool(d["is_duplicate"])))
    return pairs


def evaluate(
    pairs: list[tuple[str, str, bool]],
    embedder: Embedder,
    thresholds: list[float],
    *,
    batch_size: int = 256,
) -> list[ThresholdRow]:
    negs = RuntimeConfig().negation_terms
    texts = sorted({t for a, b, _ in pairs for t in (a, b)})
    vecs = np.concatenate(
        [embedder.embed(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
    )
    index = {t: i for i, t in enumerate(texts)}
    sims = np.array([float(vecs[index[a]] @ vecs[index[b]]) for a, b, _ in pairs])
    labels = np.array([d for _, _, d in pairs])
    guard_ok = np.array([query_signature(a, negs) == query_signature(b, negs) for a, b, _ in pairs])
    rows: list[ThresholdRow] = []
    for guard in (False, True):
        for t in thresholds:
            pred = (sims >= t) & (guard_ok if guard else True)
            tp = int((pred & labels).sum())
            fp = int((pred & ~labels).sum())
            fn = int((~pred & labels).sum())
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append(ThresholdRow(t, guard, precision, recall, f1, tp + fp, fp))
    return rows


def recommend(rows: list[ThresholdRow], min_precision: float) -> ThresholdRow | None:
    ok = [r for r in rows if r.guard and r.precision >= min_precision]
    return max(ok, key=lambda r: (r.recall, r.threshold)) if ok else None
