import pytest
from src.pipeline.pipeline import ResearchGuard
from src.schemas.query import Query

def test_researchguard_initialization():
    rg = ResearchGuard()
    assert rg.pipeline is not None
    assert rg.pipeline.retriever is not None
    assert rg.pipeline.generator is not None

def test_researchguard_empty_query():
    rg = ResearchGuard()
    with pytest.raises(ValueError):
        rg.run("")
        
def test_researchguard_missing_index():
    # If the index is empty, retriever should still handle it, or raise an error if FAISS is empty
    rg = ResearchGuard()
    # Mock retrieve to return empty or let it naturally return empty
    res = rg.run("test query")
    # depending on generator, it might generate a hallucination or say "I don't know"
    assert res is not None

def test_researchguard_invalid_k():
    rg = ResearchGuard()
    # Wait, ResearchGuard.run doesn't take k. It uses initial_k = 5.
    # To test invalid k, we can test retriever directly
    with pytest.raises(ValueError):
        rg.pipeline.retriever.retrieve("test", k=-1)
