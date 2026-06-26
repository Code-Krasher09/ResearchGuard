import pytest
from src.schemas.claim import Claim
from src.schemas.verification import VerificationResult
from src.verification.judge import Judge

@pytest.fixture
def judge():
    return Judge(threshold=0.8)

def test_judge_perfect_support(judge):
    claims = [Claim(id="1", text="claim", position=0)]
    results = [
        VerificationResult(claim_id="1", claim="claim", evidence_chunk_ids=[], top_evidence="", label="SUPPORTED", top_confidence=0.9)
    ]
    judgment = judge.judge(claims, results)
    assert judgment.faithfulness_score == 1.0
    assert not judgment.repair_needed
    assert judgment.reason == "Faithful"
    assert judgment.supported == 1
    assert judgment.total_claims == 1

def test_judge_contradiction(judge):
    claims = [Claim(id="1", text="claim", position=0)]
    results = [
        VerificationResult(claim_id="1", claim="claim", evidence_chunk_ids=[], top_evidence="", label="CONTRADICTED", top_confidence=0.9)
    ]
    judgment = judge.judge(claims, results)
    assert judgment.faithfulness_score == 0.0
    assert judgment.repair_needed
    assert judgment.reason == "Contradiction found"

def test_judge_low_support(judge):
    claims = [Claim(id=str(i), text="claim", position=i) for i in range(10)]
    results = [
        VerificationResult(claim_id=str(i), claim="claim", evidence_chunk_ids=[], top_evidence="", label="SUPPORTED" if i < 5 else "NEUTRAL", top_confidence=0.9)
        for i in range(10)
    ]
    judgment = judge.judge(claims, results)
    # 5 supported, 5 neutral
    # Score = (5 + 0.5 * 5) / 10 = 0.75
    assert judgment.faithfulness_score == 0.75  # < 0.8
    assert judgment.repair_needed
    assert judgment.reason == "Low support"

def test_judge_empty(judge):
    judgment = judge.judge([], [])
    assert judgment.repair_needed
    assert judgment.reason == "Empty claims or verification results"
    assert judgment.faithfulness_score == 0.0

def test_judge_threshold_validation():
    with pytest.raises(ValueError):
        Judge(threshold=1.5)
    with pytest.raises(ValueError):
        Judge(threshold=-0.1)
    # Should not raise
    Judge(threshold=0.5)
