import pytest
from pydantic import ValidationError

from ragcache import runtime_config


def test_defaults_are_valid_and_load_from_empty_db(db):
    cfg = runtime_config.load(db.conn)
    assert cfg.cache_similarity_threshold == 0.95
    assert "{abstain_token}" in cfg.system_prompt


def test_update_persists_only_changed_keys_and_roundtrips(db):
    runtime_config.update(db.conn, {"cache_similarity_threshold": 0.88, "top_k": 3})
    assert runtime_config.load(db.conn).top_k == 3
    keys = {r["key"] for r in db.conn.execute("SELECT key FROM runtime_config")}
    assert keys == {"cache_similarity_threshold", "top_k"}


@pytest.mark.parametrize(
    "change",
    [
        {"cache_similarity_threshold": 1.5},
        {"top_k": 0},
        {"multi_hop_patterns": ["(unclosed"]},
        {"system_prompt": "no placeholder here"},
    ],
)
def test_invalid_values_are_rejected_and_not_persisted(db, change):
    with pytest.raises((ValidationError, ValueError)):
        runtime_config.update(db.conn, change)
    assert db.conn.execute("SELECT COUNT(*) c FROM runtime_config").fetchone()["c"] == 0


def test_unknown_key_rejected(db):
    with pytest.raises(ValueError, match="unknown config keys"):
        runtime_config.update(db.conn, {"nope": 1})
