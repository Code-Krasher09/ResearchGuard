import pytest
import time
from src.generation.generator import Generator
from src.schemas.chunk import RetrievedChunk
from src.schemas.answer import GeneratedAnswer

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content="Mocked Answer"):
        self.choices = [MockChoice(content)]

@pytest.fixture
def mock_groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("MODEL_NAME", "test_model")
    monkeypatch.setattr(time, "sleep", lambda x: None)

def test_generate_answer(mock_groq_env, monkeypatch):
    generator = Generator()
    
    def mock_create(*args, **kwargs):
        # Simulate tiny network delay for latency measurement
        time.sleep(0.01) 
        return MockResponse("This is the answer.")
        
    monkeypatch.setattr(generator.client.client.chat.completions, "create", mock_create)
    
    chunks = [RetrievedChunk(chunk_id="1", text="Evidence.", source="doc", score=0.9)]
    
    result = generator.generate_answer("Query?", chunks)
    
    assert isinstance(result, GeneratedAnswer)
    assert result.answer == "This is the answer."
    assert result.model == "test_model"
    assert result.prompt_version == "v1"
    assert result.latency > 0.0
    assert result.estimated_tokens > 0

def test_generate_empty_context(mock_groq_env, monkeypatch):
    generator = Generator()
    
    def mock_create(*args, **kwargs):
        messages = kwargs["messages"]
        prompt = messages[0]["content"]
        assert "EVIDENCE:\n\n" in prompt
        return MockResponse("Insufficient info.")
        
    monkeypatch.setattr(generator.client.client.chat.completions, "create", mock_create)
    
    result = generator.generate_answer("Query?", [])
    assert result.answer == "Insufficient info."

def test_malformed_response(mock_groq_env, monkeypatch):
    generator = Generator()
    
    def mock_create(*args, **kwargs):
        raise RuntimeError("API failure")
        
    monkeypatch.setattr(generator.client.client.chat.completions, "create", mock_create)
    
    with pytest.raises(RuntimeError):
        generator.generate_answer("Query?", [])
