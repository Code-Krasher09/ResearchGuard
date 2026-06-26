import pytest
import pickle
from src.retrieval.retriever import Retriever

def test_lora_retrieval_smoke_test():
    pkl_path = "data/processed/lora_chunks.pkl"
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
        
    retriever = Retriever()
    retriever.build_from_chunks(chunks)
    
    # Query known to be present
    results = retriever.retrieve("What is LoRA?", k=3)
    
    assert len(results) == 3
    # First result should have a high score > 0.5 for bge-small on this exact text match
    assert results[0].score > 0.4
    
    # Check that metadata carried over successfully
    assert hasattr(results[0], "section")
    assert hasattr(results[0], "page")
    
    # Ensure no empty text blocks returned
    for r in results:
        assert len(r.text) > 10
