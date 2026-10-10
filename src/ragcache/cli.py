"""Command-line interface: ragcache <command>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import models, runtime_config
from .config import get_settings
from .db import Database
from .embeddings import FastEmbedEmbedder
from .retrieval import ingest_document


def _db() -> Database:
    return Database(get_settings().db_path)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("ragcache.api.app:create_app", factory=True, host=args.host, port=args.port)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    db = _db()
    if args.action == "add":
        models.add_model(
            db.conn,
            name=args.name,
            provider=args.provider,
            model_id=args.model_id,
            tier=args.tier,
            input_price_per_mtok=args.input_price,
            output_price_per_mtok=args.output_price,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            priority=args.priority,
        )
        print(f"model {args.name!r} saved")
    elif args.action == "disable":
        models.set_active(db.conn, args.name, False)
        print(f"model {args.name!r} disabled")
    else:
        for m in models.list_models(db.conn):
            print(json.dumps(m.__dict__))
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    db = _db()
    if args.action == "show":
        print(json.dumps(runtime_config.load(db.conn).model_dump(), indent=2))
    else:
        cfg = runtime_config.update(db.conn, {args.key: json.loads(args.value)})
        print(json.dumps(getattr(cfg, args.key), indent=2))
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    db = Database(settings.db_path)
    cfg = runtime_config.load(db.conn)
    embedder = FastEmbedEmbedder(settings.embedding_model)
    changed = skipped = 0
    for p in args.paths:
        path = Path(p)
        files = (
            [f for f in sorted(path.rglob("*")) if f.suffix in {".txt", ".md"}]
            if path.is_dir()
            else [path]
        )
        for f in files:
            did = ingest_document(
                db.conn,
                embedder,
                collection=args.collection,
                external_id=str(f),
                title=f.stem,
                text=f.read_text(encoding="utf-8"),
                chunk_chars=cfg.chunk_chars,
                overlap=cfg.chunk_overlap_chars,
            )
            changed += did
            skipped += not did
    print(f"ingested {changed} document(s), {skipped} unchanged")
    return 0


def _cmd_eval_threshold(args: argparse.Namespace) -> int:
    from .eval_threshold import evaluate, load_pairs, recommend

    pairs = load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    embedder = FastEmbedEmbedder(get_settings().embedding_model)
    thresholds = [round(x, 3) for x in _frange(args.start, args.stop, args.step)]
    rows = evaluate(pairs, embedder, thresholds)
    print(f"{len(pairs)} labelled pairs, {sum(1 for p in pairs if p[2])} duplicates\n")
    print("guard threshold precision recall    f1  served false_hits")
    for r in rows:
        print(
            f"{'on ' if r.guard else 'off'}   {r.threshold:9.3f} {r.precision:9.3f} "
            f"{r.recall:6.3f} {r.f1:5.3f} {r.hits:7d} {r.false_hits:10d}"
        )
    best = recommend(rows, args.min_precision)
    if best:
        print(
            f"\nrecommended (guard on, precision >= {args.min_precision}): "
            f"threshold={best.threshold} precision={best.precision:.3f} recall={best.recall:.3f}"
        )
    else:
        print(f"\nno threshold reaches precision >= {args.min_precision}")
    return 0


def _cmd_eval_verifier(args: argparse.Namespace) -> int:
    from .eval_threshold import evaluate_verifier, load_pairs
    from .verifier import CrossEncoderVerifier

    pairs = load_pairs(args.pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    embedder = FastEmbedEmbedder(get_settings().embedding_model)
    rows = evaluate_verifier(
        pairs, embedder, CrossEncoderVerifier(args.model), args.candidate, args.min_scores
    )
    print(f"{len(pairs)} labelled pairs, {sum(1 for p in pairs if p[2])} duplicates")
    print(f"candidates: similarity >= {args.candidate} and guard ok; verifier: {args.model}\n")
    print("min_score precision recall    f1  served false_hits")
    for r in rows:
        print(
            f"{r.min_score:9.2f} {r.precision:9.3f} {r.recall:6.3f} {r.f1:5.3f} "
            f"{r.hits:7d} {r.false_hits:10d}"
        )
    return 0


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, x = [], start
    while x <= stop + 1e-9:
        out.append(x)
        x += step
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=get_settings().log_level)
    p = argparse.ArgumentParser(prog="ragcache")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_cmd_serve)

    m = sub.add_parser("models", help="manage the model registry")
    msub = m.add_subparsers(dest="action", required=True)
    add = msub.add_parser("add")
    add.add_argument("name")
    add.add_argument("--provider", required=True, choices=["anthropic", "openai_compat"])
    add.add_argument("--model-id", required=True)
    add.add_argument("--tier", required=True, choices=["small", "large"])
    add.add_argument(
        "--input-price", type=float, required=True, help="USD per million input tokens"
    )
    add.add_argument(
        "--output-price", type=float, required=True, help="USD per million output tokens"
    )
    add.add_argument("--base-url")
    add.add_argument("--api-key-env", help="name of the env variable holding the key")
    add.add_argument("--priority", type=int, default=100)
    dis = msub.add_parser("disable")
    dis.add_argument("name")
    msub.add_parser("list")
    m.set_defaults(func=_cmd_models)

    c = sub.add_parser("config", help="show or change runtime configuration")
    csub = c.add_subparsers(dest="action", required=True)
    csub.add_parser("show")
    cset = csub.add_parser("set")
    cset.add_argument("key")
    cset.add_argument("value", help="JSON value, e.g. 0.9 or true")
    c.set_defaults(func=_cmd_config)

    i = sub.add_parser("ingest", help="ingest .txt/.md files into a collection")
    i.add_argument("collection")
    i.add_argument("paths", nargs="+")
    i.set_defaults(func=_cmd_ingest)

    e = sub.add_parser("eval-threshold", help="evaluate cache thresholds on labelled query pairs")
    e.add_argument("pairs", help="JSONL with q1, q2, is_duplicate")
    e.add_argument("--limit", type=int)
    e.add_argument("--start", type=float, default=0.80)
    e.add_argument("--stop", type=float, default=0.98)
    e.add_argument("--step", type=float, default=0.02)
    e.add_argument("--min-precision", type=float, default=0.95)
    e.set_defaults(func=_cmd_eval_threshold)

    v = sub.add_parser("eval-verifier", help="evaluate the cross-encoder hit verifier on labelled pairs")
    v.add_argument("pairs", help="JSONL with q1, q2, is_duplicate")
    v.add_argument("--limit", type=int)
    v.add_argument("--candidate", type=float, default=0.88)
    v.add_argument("--min-scores", type=float, nargs="+", default=[0.5, 0.7, 0.9, 0.95, 0.99])
    v.add_argument("--model", default="cross-encoder/quora-distilroberta-base")
    v.set_defaults(func=_cmd_eval_verifier)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
