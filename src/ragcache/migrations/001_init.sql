CREATE TABLE collections (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    corpus_version INTEGER NOT NULL DEFAULT 1,
    created_at     REAL NOT NULL
);

CREATE TABLE documents (
    id            INTEGER PRIMARY KEY,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    external_id   TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL,
    created_at    REAL NOT NULL,
    UNIQUE (collection_id, external_id)
);

CREATE TABLE chunks (
    id            INTEGER PRIMARY KEY,
    document_id   INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    text          TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX idx_chunks_collection ON chunks(collection_id);

CREATE TABLE cache_entries (
    id                  INTEGER PRIMARY KEY,
    collection_id       INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    corpus_version      INTEGER NOT NULL,
    query_text          TEXT NOT NULL,
    query_signature     TEXT NOT NULL,
    embedding           BLOB NOT NULL,
    answer              TEXT NOT NULL,
    citations_json      TEXT NOT NULL,
    source_model        TEXT NOT NULL,
    original_cost_usd   REAL NOT NULL,
    original_latency_ms REAL NOT NULL,
    created_at          REAL NOT NULL,
    expires_at          REAL NOT NULL,
    last_hit_at         REAL NOT NULL,
    hit_count           INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_cache_ns ON cache_entries(collection_id, corpus_version);

CREATE TABLE models (
    id                    INTEGER PRIMARY KEY,
    name                  TEXT NOT NULL UNIQUE,
    provider              TEXT NOT NULL CHECK (provider IN ('anthropic', 'openai_compat')),
    model_id              TEXT NOT NULL,
    tier                  TEXT NOT NULL CHECK (tier IN ('small', 'large')),
    base_url              TEXT,
    api_key_env           TEXT,
    input_price_per_mtok  REAL NOT NULL CHECK (input_price_per_mtok >= 0),
    output_price_per_mtok REAL NOT NULL CHECK (output_price_per_mtok >= 0),
    priority              INTEGER NOT NULL DEFAULT 100,
    active                INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE requests (
    id             INTEGER PRIMARY KEY,
    request_id     TEXT NOT NULL UNIQUE,
    ts             REAL NOT NULL,
    collection_id  INTEGER REFERENCES collections(id) ON DELETE SET NULL,
    query_hash     TEXT NOT NULL,
    query_text     TEXT,
    status         TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    cache_hit      INTEGER NOT NULL DEFAULT 0,
    similarity     REAL,
    route_tier     TEXT,
    model_name     TEXT,
    escalated      INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    saved_cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms     REAL NOT NULL,
    retrieval_ms   REAL NOT NULL DEFAULT 0,
    generation_ms  REAL NOT NULL DEFAULT 0,
    error          TEXT
);
CREATE INDEX idx_requests_ts ON requests(ts);

CREATE TABLE runtime_config (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);
