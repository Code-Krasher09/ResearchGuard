import pytest
import os
from src.retrieval.retriever import Retriever
from src.schemas.chunk import RetrievedChunk

@pytest.fixture
def retriever(tmp_path):
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.pkl")
    return Retriever(index_path=index_path, metadata_path=metadata_path, chunk_size=2, overlap=0)

def test_build_from_documents(retriever):
    docs = [
        {"text": "This is doc1 sentence 1. This is doc1 sentence 2.", "source": "doc1"},
        {"text": "This is doc2 sentence 1.", "source": "doc2"}
    ]
    retriever.build_from_documents(docs)
    
    assert retriever.store.index is not None
    assert retriever.store.index.ntotal == 2

def test_build_from_chunks(retriever):
    chunks = [
        RetrievedChunk(chunk_id="1", text="Chunk A", source="s1", score=0.0),
        RetrievedChunk(chunk_id="2", text="Chunk B", source="s2", score=0.0)
    ]
    retriever.build_from_chunks(chunks)
    assert retriever.store.index.ntotal == 2
    assert chunks[0].embedding is not None

def test_retrieve_and_search(retriever):
    docs = [{"text": "Machine learning involves models.", "source": "ML"}]
    retriever.build_from_documents(docs)
    
    results = retriever.retrieve("What involves models?", k=1)
    assert len(results) == 1
    assert results[0].source == "ML"
    assert "Machine learning" in results[0].text
    
    # search() alias
    results2 = retriever.search("What involves models?", k=1)
    assert len(results2) == 1
    assert results[0].chunk_id == results2[0].chunk_id

def test_retrieve_invalid(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("", k=1)
    with pytest.raises(ValueError):
        retriever.retrieve("Query", k=0)

def test_save_load(retriever, tmp_path):
    docs = [{"text": "Persist this.", "source": "src"}]
    retriever.build_from_documents(docs)
    retriever.save()
    
    index_path = str(tmp_path / "index.faiss")
    metadata_path = str(tmp_path / "metadata.pkl")
    
    retriever2 = Retriever(index_path=index_path, metadata_path=metadata_path)
    retriever2.load_index()
    
    assert retriever2.store.index.ntotal == 1
    res = retriever2.retrieve("Persist", k=1)
    assert len(res) == 1
    assert res[0].source == "src"
