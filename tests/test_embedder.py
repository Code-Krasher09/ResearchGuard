import pytest
import math
from src.retrieval.embedder import Embedder, MODEL_CACHE
from src.schemas.chunk import RetrievedChunk

def test_embedder_cache():
    # Clear cache for isolated test
    MODEL_CACHE.clear()
    e1 = Embedder("BAAI/bge-small-en-v1.5")
    e2 = Embedder("BAAI/bge-small-en-v1.5")
    assert e1._model is e2._model
    assert len(MODEL_CACHE) == 1

def test_get_dimension():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    dim = embedder.get_dimension()
    assert isinstance(dim, int)
    assert dim > 0

def test_embed_query_empty():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    with pytest.raises(ValueError):
        embedder.embed_query("   ")

def test_embed_query():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    query = "What is LoRA?"
    emb = embedder.embed_query(query)
    
    assert isinstance(emb, list)
    assert len(emb) == embedder.get_dimension()
    assert isinstance(emb[0], float)

def test_normalization():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    emb = embedder.embed_query("Normalize this.")
    
    l2_norm = math.sqrt(sum(x*x for x in emb))
    assert abs(l2_norm - 1.0) < 1e-3

def test_embed_chunks():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    
    chunks = [
        RetrievedChunk(chunk_id="1", text="Sentence one.", source="doc1", score=0.0),
        RetrievedChunk(chunk_id="2", text="Sentence two.", source="doc2", score=0.0),
        RetrievedChunk(chunk_id="3", text="Sentence three.", source="doc3", score=0.0)
    ]
    
    embeddings = embedder.embed_chunks(chunks, batch_size=2)
    
    assert len(embeddings) == 3
    assert len(embeddings[0]) == embedder.get_dimension()
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) == embedder.get_dimension()

def test_embed_empty_chunks():
    embedder = Embedder("BAAI/bge-small-en-v1.5")
    embeddings = embedder.embed_chunks([])
    assert embeddings == []
