import pytest
from src.generation.generator import Generator
from src.generation.prompts import PromptManager
from src.schemas.chunk import RetrievedChunk

@pytest.fixture
def generator():
    prompt_manager = PromptManager()
    return Generator(prompt_manager=prompt_manager)

def test_transformer_question_grounded(generator):
    context = [
        RetrievedChunk(chunk_id="12", text="The Transformer is a sequence transduction model that relies entirely on self-attention.", source="Attention_is_All_You_Need.pdf", score=0.9),
        RetrievedChunk(chunk_id="13", text="The Transformer follows an encoder-decoder architecture.", source="Attention_is_All_You_Need.pdf", score=0.85)
    ]
    query = "What is a Transformer?"
    
    answer_obj = generator.generate_answer(query, context, mode="grounded_extraction")
    answer = answer_obj.answer.lower()
    
    # Grounded extraction should synthesize a response, not reject it.
    assert "insufficient evidence" not in answer
    assert "sequence transduction" in answer or "self-attention" in answer
    assert "encoder" in answer or "decoder" in answer
    # Check that chunk citation is included
    assert "12" in answer or "13" in answer

def test_lora_question_unanswerable(generator):
    context = [
        RetrievedChunk(chunk_id="12", text="The Transformer is a sequence transduction model that relies entirely on self-attention.", source="Attention_is_All_You_Need.pdf", score=0.9),
        RetrievedChunk(chunk_id="13", text="The Transformer follows an encoder-decoder architecture.", source="Attention_is_All_You_Need.pdf", score=0.85)
    ]
    query = "Who invented LoRA?"
    
    answer_obj = generator.generate_answer(query, context, mode="grounded_extraction")
    
    assert answer_obj.answer.strip().upper() == "INSUFFICIENT EVIDENCE"

def test_france_question_unanswerable(generator):
    context = [
        RetrievedChunk(chunk_id="1", text="The study investigates asthma in children.", source="doc1.pdf", score=0.9)
    ]
    query = "What is the capital of France?"
    
    answer_obj = generator.generate_answer(query, context, mode="grounded_extraction")
    
    assert answer_obj.answer.strip().upper() == "INSUFFICIENT EVIDENCE"

def test_empty_context(generator):
    context = []
    query = "What is asthma?"
    
    answer_obj = generator.generate_answer(query, context, mode="grounded_extraction")
    
    assert answer_obj.answer.strip().upper() == "INSUFFICIENT EVIDENCE"
