import pytest
from src.verification.claims import ClaimExtractor
from src.schemas.claim import Claim

@pytest.fixture
def extractor():
    return ClaimExtractor()

def test_extract_claims_basic(extractor):
    answer = "LoRA freezes pretrained weights. LoRA trains low rank matrices."
    claims = extractor.extract_claims(answer)
    assert len(claims) == 2
    assert claims[0].text == "LoRA freezes pretrained weights."
    assert claims[1].text == "LoRA trains low rank matrices."
    assert claims[0].position == 0
    assert claims[1].id == "claim_0001"

def test_conversational_removal(extractor):
    answer = "I think LoRA freezes pretrained weights. Perhaps it also trains matrices."
    claims = extractor.extract_claims(answer)
    assert len(claims) == 2
    assert claims[0].text == "LoRA freezes pretrained weights."
    assert claims[1].text == "It also trains matrices."

def test_empty_and_short(extractor):
    answer = "I think... OK. LoRA is good."
    claims = extractor.extract_claims(answer)
    assert len(claims) == 1
    assert claims[0].text == "LoRA is good."

def test_extract_json_valid(extractor):
    json_text = '[{"id": "1", "text": "Claim 1"}, {"id": "2", "text": "Claim 2"}]'
    claims = extractor.extract_json(json_text)
    assert len(claims) == 2
    assert claims[0].text == "Claim 1"

def test_extract_json_invalid(extractor):
    claims = extractor.extract_json("Not JSON")
    assert len(claims) == 0

def test_large_answer(extractor):
    answer = "Sentence one. " * 100
    claims = extractor.extract_claims(answer)
    assert len(claims) == 100
