import os
import json

def test_lora_corpus_exists():
    assert os.path.exists("data/processed/lora_chunks.json")
    assert os.path.exists("data/processed/lora_chunks.pkl")

def test_lora_corpus_chunk_count():
    with open("data/processed/lora_chunks.json", "r") as f:
        chunks = json.load(f)
    
    assert len(chunks) > 0, "Corpus is empty"
    assert 20 <= len(chunks) <= 100, f"Expected 20-100 chunks, got {len(chunks)}"

def test_lora_corpus_serialization():
    with open("data/processed/lora_chunks.json", "r") as f:
        chunks = json.load(f)
        
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk
        assert "section" in chunk
        assert "page" in chunk
        
        assert len(chunk["text"]) >= 10, "Chunk text is too short"
        assert chunk["source"] == "lora.pdf"
