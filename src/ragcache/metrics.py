"""Aggregate the request log into the numbers shown on the dashboard."""

from __future__ import annotations

import sqlite3
import time

import numpy as np


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _latency(values: list[float]) -> dict[str, float | None]:
    return {"p50": _pct(values, 50), "p95": _pct(values, 95), "p99": _pct(values, 99)}


def summary(conn: sqlite3.Connection, window_seconds: float | None = None) -> dict[str, object]:
    since = time.time() - window_seconds if window_seconds else 0.0
    rows = conn.execute(
        "SELECT status, cache_hit, route_tier, escalated, input_tokens, output_tokens, cost_usd, "
        "saved_cost_usd, latency_ms FROM requests WHERE ts >= ?",
        (since,),
    ).fetchall()
    ok = [r for r in rows if r["status"] == "ok"]
    hits = [r for r in ok if r["cache_hit"]]
    misses = [r for r in ok if not r["cache_hit"]]
    by_route: dict[str, dict[str, object]] = {}
    for label, group in (
        ("cache_hit", hits),
        ("small", [r for r in misses if r["route_tier"] == "small"]),
        ("large", [r for r in misses if r["route_tier"] == "large"]),
    ):
        lat = [r["latency_ms"] for r in group]
        by_route[label] = {
            "requests": len(group),
            "cost_usd": sum(r["cost_usd"] for r in group),
            "latency_ms": _latency(lat),
        }
    total_cost = sum(r["cost_usd"] for r in ok)
    saved = sum(r["saved_cost_usd"] for r in hits)
    return {
        "window_seconds": window_seconds,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "cache_hit_rate": (len(hits) / len(ok)) if ok else None,
        "escalation_rate": (
            sum(1 for r in misses if r["escalated"]) / len(misses) if misses else None
        ),
        "tokens": {
            "input": sum(r["input_tokens"] for r in ok),
            "output": sum(r["output_tokens"] for r in ok),
        },
        "cost_usd": total_cost,
        "saved_cost_usd": saved,
        "latency_ms": _latency([r["latency_ms"] for r in ok]),
        "by_route": by_route,
    }
