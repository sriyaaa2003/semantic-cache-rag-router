"""Deterministic, configurable query router: decides whether a query needs the small or large tier.

The score is a weighted sum of cheap, inspectable features. Weights, threshold and the multi-hop
patterns are runtime configuration, so they can be re-tuned from logged outcomes without a deploy.
A learned router is deliberately not used: it needs labelled outcome data that does not exist
until the system has run, and a transparent rule is easier to audit and debug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .runtime_config import RuntimeConfig


@dataclass(frozen=True)
class RouteDecision:
    tier: str  # "small" | "large"
    score: float
    reasons: list[str] = field(default_factory=list)


def decide(query: str, top_retrieval_score: float | None, cfg: RuntimeConfig) -> RouteDecision:
    w = cfg.router_weights
    reasons: list[str] = []
    score = 0.0

    tokens = len(query.split())
    length_feature = min(tokens / cfg.router_length_reference_tokens, 1.0)
    score += w.length * length_feature
    if length_feature >= 0.5:
        reasons.append(f"long query ({tokens} tokens)")

    if any(re.search(p, query, re.IGNORECASE) for p in cfg.multi_hop_patterns):
        score += w.multi_hop_cue
        reasons.append("comparison / multi-part cue")

    if top_retrieval_score is None:
        score += w.low_retrieval_confidence
        reasons.append("no retrieved context")
    else:
        # 1.0 when retrieval is weak (cosine <= 0.5), 0.0 when strong (cosine >= 0.85)
        weak = min(max((0.85 - top_retrieval_score) / 0.35, 0.0), 1.0)
        score += w.low_retrieval_confidence * weak
        if weak >= 0.5:
            reasons.append(f"weak retrieval (top score {top_retrieval_score:.2f})")

    clauses = len(re.findall(r"[,;?]| and | or ", query))
    if clauses >= 3:
        score += w.many_clauses
        reasons.append(f"{clauses} clauses")

    tier = "large" if score >= cfg.router_large_threshold else "small"
    if not reasons:
        reasons.append("simple, well-supported query")
    return RouteDecision(tier=tier, score=round(score, 4), reasons=reasons)
