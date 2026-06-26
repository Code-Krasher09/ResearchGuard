import pytest
import datetime
from pydantic import ValidationError
from src.schemas.query import Query
from src.schemas.chunk import RetrievedChunk

def test_query_valid():
    q = Query.create("What is LoRA?", "q123")
    assert q.id == "q123"
    assert q.text == "What is LoRA?"
    assert isinstance(q.timestamp, datetime.datetime)
    assert q.metadata == {}

    q2 = Query(id="q124", text="Test", timestamp="2026-06-25T12:00:00Z", metadata={"key": "value"})
    assert q2.metadata["key"] == "value"

def test_query_create_no_id_no_metadata():
    q = Query.create(text="What is LoRA?", metadata=None)
    assert q.id is not None
    assert isinstance(q.id, str)
    assert len(q.id) > 10
    assert q.metadata == {}

def test_query_invalid():
    with pytest.raises(ValidationError):
        Query(id="q123", text="", timestamp="2026-06-25T12:00:00Z")  # text too short
        
    with pytest.raises(ValidationError):
        Query(text="Missing id", timestamp="2026-06-25T12:00:00Z")
        
    with pytest.raises(ValidationError):
        Query(id="q125", text="Invalid time", timestamp="Not a timestamp")

def test_retrieved_chunk_valid():
    chunk = RetrievedChunk(
        chunk_id="c123",
        text="Sample text.",
        source="paper_x",
        score=0.95
    )
    assert chunk.chunk_id == "c123"
    assert chunk.score == 0.95
    assert chunk.embedding is None

def test_retrieved_chunk_invalid():
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk_id="c123", text="", source="paper_x", score=0.95) # text too short

    with pytest.raises(ValidationError):
        RetrievedChunk(chunk_id="c123", text="text", source="paper_x", score=1.5) # score > 1

    with pytest.raises(ValidationError):
        RetrievedChunk(chunk_id="c123", text="text", source="paper_x", score=-1.5) # score < -1
