import pytest

from ragcache.embeddings import FastEmbedEmbedder

pytestmark = pytest.mark.embedding


def test_real_model_separates_paraphrase_from_unrelated():
    emb = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    v = emb.embed(["How do I reset my password?", "password reset steps", "best pizza in Rome"])
    assert v.shape == (3, 384)
    assert float(v[0] @ v[1]) > 0.7 > float(v[0] @ v[2])
