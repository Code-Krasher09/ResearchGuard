import pytest
from unittest.mock import MagicMock
from src.schemas.claim import Claim
from src.schemas.chunk import RetrievedChunk
from src.schemas.verification import VerificationResult
import src.verification.verifier as verifier_mod

@pytest.fixture
def mock_verifier(monkeypatch):
    mock_torch = MagicMock()
    mock_torch.device = MagicMock(return_value="cpu")
    mock_torch.no_grad = MagicMock(return_value=MagicMock())
    mock_torch.softmax = MagicMock()
    
    mock_tokenizer = MagicMock()
    mock_model_cls = MagicMock()
    
    monkeypatch.setattr(verifier_mod, "torch", mock_torch)
    monkeypatch.setattr(verifier_mod, "AutoTokenizer", mock_tokenizer)
    monkeypatch.setattr(verifier_mod, "AutoModelForSequenceClassification", mock_model_cls)
    
    verifier = verifier_mod.Verifier()
    verifier._mock_torch = mock_torch
    return verifier

def test_verify_empty_evidence(mock_verifier):
    claim = Claim(id="claim_0000", text="Sky is blue.", position=0)
    result = mock_verifier.verify_claim(claim, [])
    assert result.label == "NEUTRAL"
    assert result.top_confidence == 1.0
    assert result.evidence_chunk_ids == []

def test_verify_batch_empty_claims(mock_verifier):
    results = mock_verifier.verify_batch([], [])
    assert results == []

def test_verify_supported(mock_verifier):
    claim = Claim(id="claim_0000", text="Sky is blue.", position=0)
    evidence = [RetrievedChunk(chunk_id="chk_1", text="The sky is blue.", source="doc", score=1.0)]
    
    mock_verifier.model.config.id2label = {0: "entailment"}
    mock_probs = MagicMock()
    mock_probs.argmax.return_value.tolist.return_value = [0]
    mock_probs.max.return_value.values.tolist.return_value = [0.99]
    mock_verifier._mock_torch.softmax.return_value = mock_probs
    
    result = mock_verifier.verify_claim(claim, evidence)
    assert result.label == "SUPPORTED"
    assert result.top_confidence == 0.99
    assert result.evidence_chunk_ids == ["chk_1"]
    assert result.top_evidence == "The sky is blue."

def test_verify_contradicted(mock_verifier):
    claim = Claim(id="claim_0000", text="Sky is green.", position=0)
    evidence = [RetrievedChunk(chunk_id="chk_1", text="The sky is blue.", source="doc", score=1.0)]
    
    mock_verifier.model.config.id2label = {2: "contradiction"}
    mock_probs = MagicMock()
    mock_probs.argmax.return_value.tolist.return_value = [2]
    mock_probs.max.return_value.values.tolist.return_value = [0.95]
    mock_verifier._mock_torch.softmax.return_value = mock_probs
    
    result = mock_verifier.verify_claim(claim, evidence)
    assert result.label == "CONTRADICTED"
    assert result.top_confidence == 0.95

def test_verify_neutral(mock_verifier):
    claim = Claim(id="claim_0000", text="Water is wet.", position=0)
    evidence = [RetrievedChunk(chunk_id="chk_1", text="The sky is blue.", source="doc", score=1.0)]
    
    mock_verifier.model.config.id2label = {1: "neutral"}
    mock_probs = MagicMock()
    mock_probs.argmax.return_value.tolist.return_value = [1]
    mock_probs.max.return_value.values.tolist.return_value = [0.60]
    mock_verifier._mock_torch.softmax.return_value = mock_probs
    
    result = mock_verifier.verify_claim(claim, evidence)
    assert result.label == "NEUTRAL"
    assert result.top_confidence == 0.60
