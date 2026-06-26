import pytest
from src.schemas.claim import Claim
from src.schemas.verification import VerificationResult
from src.verification.claims import ClaimExtractor
from src.verification.judge import Judge

def test_safe_refusal_extraction():
    extractor = ClaimExtractor(min_claim_length=5)
    
    # 1. Pure refusal
    claims = extractor.extract_claims("I do not have enough information to answer.")
    assert len(claims) == 1
    assert claims[0].claim_type == "REFUSAL"
    
    # 2. Capital of France from LoRA paper
    claims = extractor.extract_claims("The provided document does not mention the capital of France.")
    assert len(claims) == 1
    assert claims[0].claim_type == "REFUSAL"
    
    # 3. Mixed factual + refusal answer
    claims = extractor.extract_claims("LoRA freezes pretrained weights. I don't know what rank was used.")
    assert len(claims) == 2
    assert claims[0].claim_type == "FACTUAL"
    assert claims[1].claim_type == "REFUSAL"

def test_safe_refusal_judge():
    judge = Judge(threshold=0.8)
    
    # All claims are REFUSAL
    claims = [
        Claim(id="claim_0000", text="I don't know.", position=0, claim_type="REFUSAL"),
        Claim(id="claim_0001", text="There is insufficient evidence.", position=1, claim_type="REFUSAL")
    ]
    # In reality verifications might not even be generated for REFUSALs, 
    # but let's assume they were and returned NEUTRAL
    verifications = [
        VerificationResult(claim_id="claim_0000", claim="I don't know.", evidence_chunk_ids=[], top_evidence="", label="NEUTRAL", top_confidence=1.0),
        VerificationResult(claim_id="claim_0001", claim="There is insufficient evidence.", evidence_chunk_ids=[], top_evidence="", label="NEUTRAL", top_confidence=1.0)
    ]
    
    judgment = judge.judge(claims, verifications)
    assert judgment.repair_needed == False
    assert judgment.faithfulness_score == 1.0
    assert judgment.reason == "Safe refusal"
    assert judgment.total_claims == 2
    
    # Mixed REFUSAL and FACTUAL
    claims.append(Claim(id="claim_0002", text="LoRA reduces parameters.", position=2, claim_type="FACTUAL"))
    verifications.append(VerificationResult(claim_id="claim_0002", claim="LoRA reduces parameters.", evidence_chunk_ids=[], top_evidence="LoRA reduces params", label="SUPPORTED", top_confidence=1.0))
    
    judgment = judge.judge(claims, verifications)
    assert judgment.repair_needed == False
    assert judgment.faithfulness_score == 1.0 # 1 FACTUAL claim, SUPPORTED -> 1/1
    assert judgment.total_claims == 3
