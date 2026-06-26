import os
import pytest
import numpy as np
from src.retrieval.faiss_store import FAISSStore
from src.schemas.chunk import RetrievedChunk

@pytest.fixture
def store(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.pkl")
    return FAISSStore(dimension=3, index_path=index_path, metadata_path=metadata_path)

def test_build_index(store):
    store.build_index()
    assert store.index is not None
    assert store.index.d == 3
    assert store.index.ntotal == 0

def test_add_and_search(store):
    chunks = [
        RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0, embedding=[1.0, 0.0, 0.0]),
        RetrievedChunk(chunk_id="2", text="B", source="s", score=0.0, embedding=[0.0, 1.0, 0.0])
    ]
    store.add(chunks)
    assert store.index.ntotal == 2
    
    # Search for [1.0, 0.0, 0.0]
    results = store.search([1.0, 0.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "1"
    assert results[0].score > 0.99  # Inner product of normalized vectors = 1.0
    
def test_add_without_embedding(store):
    chunks = [RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0)]
    with pytest.raises(ValueError):
        store.add(chunks)

def test_add_wrong_dimension(store):
    chunks = [RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0, embedding=[1.0, 0.0])]
    with pytest.raises(ValueError):
        store.add(chunks)

def test_add_unnormalized(store):
    chunks = [RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0, embedding=[2.0, 0.0, 0.0])]
    with pytest.raises(ValueError):
        store.add(chunks)

def test_search_wrong_dimension(store):
    store.build_index()
    with pytest.raises(ValueError):
        store.search([1.0, 0.0])

def test_search_unnormalized(store):
    store.build_index()
    with pytest.raises(ValueError):
        store.search([2.0, 0.0, 0.0])

def test_search_invalid_k(store):
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], k=0)
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], k=-1)

def test_save_and_load(store, tmp_path):
    chunks = [
        RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0, embedding=[1.0, 0.0, 0.0])
    ]
    store.add(chunks)
    store.save()
    
    # Create new instance
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.pkl")
    store2 = FAISSStore(dimension=3, index_path=index_path, metadata_path=metadata_path)
    store2.load()
    
    assert store2.index.ntotal == 1
    assert store2._next_id == 1
    assert len(store2.metadata) == 1
    
    results = store2.search([1.0, 0.0, 0.0], k=1)
    assert results[0].chunk_id == "1"

def test_load_not_found():
    store = FAISSStore(dimension=3, index_path="invalid/path/index.faiss", metadata_path="invalid/path/meta.pkl")
    with pytest.raises(FileNotFoundError):
        store.load()

def test_load_mismatched_dimension(store, tmp_path):
    chunks = [RetrievedChunk(chunk_id="1", text="A", source="s", score=0.0, embedding=[1.0, 0.0, 0.0])]
    store.add(chunks)
    store.save()
    
    # Try to load into a store expecting dimension 4
    store2 = FAISSStore(dimension=4, index_path=store.index_path, metadata_path=store.metadata_path)
    with pytest.raises(ValueError):
        store2.load()

def test_search_empty(store):
    results = store.search([1.0, 0.0, 0.0])
    assert results == []

def test_save_empty(store):
    # Should warn and do nothing
    store.save()
    assert not os.path.exists(store.index_path)
