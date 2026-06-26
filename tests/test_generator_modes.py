import pytest
from src.generation.generator import Generator
from src.schemas.chunk import RetrievedChunk

@pytest.fixture
def generator():
    # Provide dummy client or mock it if needed
    # Wait, client actually hits Groq API.
    # To run this test, we can just use the actual Generator, but it requires a real API call.
    # I will construct a real Generator.
    return Generator()

def test_strict_extraction_mode_unsupported(generator):
    context = [
        RetrievedChunk(chunk_id="1", text="LoRA reduces parameters by training low rank matrices.", source="test", page=1, section="Method", score=1.0)
    ]
    
    # Who invented LoRA -> Not in context
    res = generator.generate_answer("Who invented LoRA?", context, mode="strict_extraction")
    assert "INSUFFICIENT EVIDENCE" in res.answer.upper(), f"Expected INSUFFICIENT EVIDENCE, got {res.answer}"

    # Capital of France -> Not in context
    res = generator.generate_answer("What is the capital of France?", context, mode="strict_extraction")
    assert "INSUFFICIENT EVIDENCE" in res.answer.upper()
    
    # Asthma question -> Not in context
    res = generator.generate_answer("What disease causes asthma?", context, mode="strict_extraction")
    assert "INSUFFICIENT EVIDENCE" in res.answer.upper()
    
def test_strict_extraction_mode_supported(generator):
    context = [
        RetrievedChunk(chunk_id="1", text="LoRA freezes pretrained weights.", source="test", page=1, section="Method", score=1.0)
    ]
    
    res = generator.generate_answer("Does LoRA freeze weights?", context, mode="strict_extraction")
    assert "INSUFFICIENT EVIDENCE" not in res.answer.upper()
    assert "freeze" in res.answer.lower() or "freezes" in res.answer.lower()
