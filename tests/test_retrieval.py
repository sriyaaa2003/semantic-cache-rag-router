import pytest

from ragcache.retrieval import chunk_text, ingest_document

from .conftest import HashEmbedder


def test_chunk_text_respects_size_and_keeps_all_content():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, 200, 30)
    assert all(len(c) <= 200 + 30 + 1 for c in chunks)
    assert len(chunks) > 5
    assert "Sentence number 0 is here." in chunks[0]
    assert "Sentence number 59 is here." in chunks[-1]


def test_chunk_text_splits_overlong_sentence():
    chunks = chunk_text("x" * 1000, 300, 50)
    assert len(chunks) >= 4 and all(len(c) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("abc. def.", 100, 100)


def test_ingest_is_idempotent_and_bumps_version_on_change(db):
    emb = HashEmbedder()
    kwargs = dict(collection="c", external_id="d1", title="t", chunk_chars=200, overlap=20)
    assert ingest_document(db.conn, emb, text="Alpha beta gamma.", **kwargs) is True
    v1 = db.conn.execute("SELECT corpus_version v FROM collections").fetchone()["v"]
    assert ingest_document(db.conn, emb, text="Alpha beta gamma.", **kwargs) is False
    assert db.conn.execute("SELECT corpus_version v FROM collections").fetchone()["v"] == v1
    assert ingest_document(db.conn, emb, text="Totally different text.", **kwargs) is True
    assert db.conn.execute("SELECT corpus_version v FROM collections").fetchone()["v"] == v1 + 1
    assert db.conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"] == 1


def test_empty_document_rejected(db):
    with pytest.raises(ValueError):
        ingest_document(
            db.conn,
            HashEmbedder(),
            collection="c",
            external_id="d",
            title="",
            text="   ",
            chunk_chars=200,
            overlap=20,
        )


def test_search_returns_most_similar_first(pipeline, corpus):
    row = pipeline.db.conn.execute("SELECT id, corpus_version v FROM collections").fetchone()
    q = pipeline.embedder.embed(["how long does shipping take"])[0]
    hits = pipeline.retriever.search(row["id"], row["v"], q, 3)
    assert hits[0].document_external_id == "shipping"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
