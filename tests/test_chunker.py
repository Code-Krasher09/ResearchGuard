import pytest
from src.retrieval.chunker import Chunker
from src.schemas.chunk import RetrievedChunk

def test_chunker_invalid_init():
    with pytest.raises(ValueError):
        Chunker(0, 1)
    with pytest.raises(ValueError):
        Chunker(5, 5)
    with pytest.raises(ValueError):
        Chunker(-1, 1)
    with pytest.raises(ValueError):
        Chunker(5, -1)

def test_chunker_initialization():
    chunker = Chunker(chunk_size=3, overlap=1)
    assert chunker.chunk_size == 3
    assert chunker.overlap == 1

def test_chunk_text_basic():
    chunker = Chunker(chunk_size=2, overlap=1)
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = chunker.chunk_text(text, source="test_doc")
    
    assert len(chunks) == 4
    for chunk in chunks:
        assert isinstance(chunk, RetrievedChunk)
        assert chunk.source == "test_doc"
        assert chunk.score == 0.0
        assert chunk.embedding is None
        
    assert "This is sentence one." in chunks[0].text
    assert "This is sentence two." in chunks[0].text
    
    # Overlap
    assert "This is sentence two." in chunks[1].text
    assert "This is sentence three." in chunks[1].text
    
    assert "This is sentence three." in chunks[2].text
    assert "This is sentence four." in chunks[2].text

    assert "This is sentence four." in chunks[3].text
    assert "This is sentence three." not in chunks[3].text

def test_chunk_text_long_document():
    chunker = Chunker(chunk_size=5, overlap=2)
    sentences = [f"This is sentence {i}." for i in range(1, 13)]
    text = " ".join(sentences)
    chunks = chunker.chunk_text(text, source="long_doc")
    
    # 12 sentences.
    # Chunk 1: 1-5 (Next starts at 1 + 5 - 2 = 4)
    # Chunk 2: 4-8 (Next starts at 4 + 5 - 2 = 7)
    # Chunk 3: 7-11 (Next starts at 7 + 5 - 2 = 10)
    # Chunk 4: 10-12
    assert len(chunks) == 4
    assert "This is sentence 1." in chunks[0].text
    assert "This is sentence 5." in chunks[0].text
    
    assert "This is sentence 4." in chunks[1].text
    assert "This is sentence 8." in chunks[1].text
    
    assert "This is sentence 7." in chunks[2].text
    assert "This is sentence 11." in chunks[2].text
    
    assert "This is sentence 10." in chunks[3].text
    assert "This is sentence 12." in chunks[3].text
    assert "This is sentence 13." not in chunks[3].text

def test_chunk_text_empty():
    chunker = Chunker()
    chunks = chunker.chunk_text("   ", "empty_doc")
    assert len(chunks) == 0

def test_chunk_text_less_than_size():
    chunker = Chunker(chunk_size=5, overlap=1)
    text = "Only one sentence."
    chunks = chunker.chunk_text(text, "short_doc")
    
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence."
