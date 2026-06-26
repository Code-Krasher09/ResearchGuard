import pytest
import time
from src.generation.client import GroqClient
from groq import APIConnectionError

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content="Mocked Response"):
        self.choices = [MockChoice(content)]

@pytest.fixture
def mock_groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    monkeypatch.setenv("MODEL_NAME", "test_model")
    monkeypatch.setenv("TEMPERATURE", "0.5")
    monkeypatch.setenv("MAX_TOKENS", "100")
    # Disable wait to speed up tests
    monkeypatch.setattr(time, "sleep", lambda x: None)

def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        GroqClient(api_key=None)

def test_client_init(mock_groq_env):
    client = GroqClient()
    assert client.api_key == "test_key"
    assert client.default_model == "test_model"
    assert client.default_temperature == 0.5
    assert client.default_max_tokens == 100

def test_generate_success(mock_groq_env, monkeypatch):
    client = GroqClient()
    
    def mock_create(*args, **kwargs):
        assert kwargs["model"] == "test_model"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100
        return MockResponse("Hello!")
        
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create)
    
    res = client.generate("Say hello")
    assert res == "Hello!"

def test_chat_success(mock_groq_env, monkeypatch):
    client = GroqClient()
    
    def mock_create(*args, **kwargs):
        return MockResponse("Hi back!")
        
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create)
    
    res = client.chat([{"role": "user", "content": "Hi"}])
    assert res == "Hi back!"

def test_chat_invalid_messages(mock_groq_env):
    client = GroqClient()
    with pytest.raises(ValueError, match="Each message must contain 'role' and 'content' keys."):
        client.chat([{"role": "user"}]) # Missing content
    with pytest.raises(ValueError, match="Each message must contain 'role' and 'content' keys."):
        client.chat([{"content": "hello"}]) # Missing role

def test_stream_param(mock_groq_env, monkeypatch):
    client = GroqClient()
    def mock_create(*args, **kwargs):
        assert kwargs["stream"] is True
        return ["chunk1", "chunk2"]
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create)
    res = client.generate("stream this", stream=True)
    assert res == ["chunk1", "chunk2"]

def test_empty_prompt(mock_groq_env):
    client = GroqClient()
    with pytest.raises(ValueError):
        client.generate("   ")
        
def test_empty_messages(mock_groq_env):
    client = GroqClient()
    with pytest.raises(ValueError):
        client.chat([])

def test_retry_behavior(mock_groq_env, monkeypatch):
    client = GroqClient()
    
    call_count = 0
    def mock_create_fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise APIConnectionError(request=None)
        return MockResponse("Finally Success")
        
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create_fail)
    
    res = client.generate("Test retry")
    assert res == "Finally Success"
    assert call_count == 3
    
def test_retry_exhausted(mock_groq_env, monkeypatch):
    client = GroqClient()
    
    def mock_create_fail(*args, **kwargs):
        raise APIConnectionError(request=None)
        
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create_fail)
    
    with pytest.raises(APIConnectionError):
        client.generate("Test retry fail")

def test_unexpected_error(mock_groq_env, monkeypatch):
    client = GroqClient()
    
    def mock_create_fail(*args, **kwargs):
        raise RuntimeError("Unexpected boom")
        
    monkeypatch.setattr(client.client.chat.completions, "create", mock_create_fail)
    
    with pytest.raises(RuntimeError):
        client.generate("Test unexpected")

def test_health_check_success(mock_groq_env, monkeypatch):
    client = GroqClient()
    def mock_list(*args, **kwargs):
        return []
    monkeypatch.setattr(client.client.models, "list", mock_list)
    assert client.health_check() is True

def test_health_check_failure(mock_groq_env, monkeypatch):
    client = GroqClient()
    def mock_list(*args, **kwargs):
        raise APIConnectionError(request=None)
    monkeypatch.setattr(client.client.models, "list", mock_list)
    assert client.health_check() is False
