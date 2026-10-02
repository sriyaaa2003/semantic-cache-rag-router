"""Model registry (database-backed) and cost accounting."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    tier: str
    base_url: str | None
    api_key_env: str | None
    input_price_per_mtok: float
    output_price_per_mtok: float
    priority: int

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_price_per_mtok + output_tokens * self.output_price_per_mtok
        ) / 1_000_000


def _row(r: sqlite3.Row) -> ModelSpec:
    return ModelSpec(
        name=r["name"],
        provider=r["provider"],
        model_id=r["model_id"],
        tier=r["tier"],
        base_url=r["base_url"],
        api_key_env=r["api_key_env"],
        input_price_per_mtok=r["input_price_per_mtok"],
        output_price_per_mtok=r["output_price_per_mtok"],
        priority=r["priority"],
    )


def add_model(
    conn: sqlite3.Connection,
    *,
    name: str,
    provider: str,
    model_id: str,
    tier: str,
    input_price_per_mtok: float,
    output_price_per_mtok: float,
    base_url: str | None = None,
    api_key_env: str | None = None,
    priority: int = 100,
) -> None:
    conn.execute(
        "INSERT INTO models(name, provider, model_id, tier, base_url, api_key_env, "
        "input_price_per_mtok, output_price_per_mtok, priority, active) "
        "VALUES (?,?,?,?,?,?,?,?,?,1) "
        "ON CONFLICT(name) DO UPDATE SET provider=excluded.provider, model_id=excluded.model_id, "
        "tier=excluded.tier, base_url=excluded.base_url, api_key_env=excluded.api_key_env, "
        "input_price_per_mtok=excluded.input_price_per_mtok, "
        "output_price_per_mtok=excluded.output_price_per_mtok, priority=excluded.priority, "
        "active=1",
        (
            name,
            provider,
            model_id,
            tier,
            base_url,
            api_key_env,
            input_price_per_mtok,
            output_price_per_mtok,
            priority,
        ),
    )


def set_active(conn: sqlite3.Connection, name: str, active: bool) -> None:
    cur = conn.execute("UPDATE models SET active=? WHERE name=?", (int(active), name))
    if cur.rowcount == 0:
        raise KeyError(name)


def list_models(conn: sqlite3.Connection, *, tier: str | None = None) -> list[ModelSpec]:
    sql = "SELECT * FROM models WHERE active=1"
    args: tuple[str, ...] = ()
    if tier:
        sql += " AND tier=?"
        args = (tier,)
    sql += " ORDER BY priority, name"
    return [_row(r) for r in conn.execute(sql, args)]
