import pytest
from src.generation.prompts import PromptManager
from src.schemas.chunk import RetrievedChunk

def test_scientific_qa():
    pm = PromptManager()
    chunks = [RetrievedChunk(chunk_id="1", text="DNA is a molecule.", source="doc1", score=0.9)]
    prompt = pm.scientific_qa("What is DNA?", chunks)
    assert "EVIDENCE:\n[doc1] DNA is a molecule." in prompt
    assert "QUESTION: What is DNA?" in prompt
    assert "Do NOT fabricate citations" in prompt

def test_scientific_qa_empty_context():
    pm = PromptManager()
    prompt = pm.scientific_qa("What is DNA?", [])
    assert "EVIDENCE:\n\n" in prompt
    assert "QUESTION: What is DNA?" in prompt

def test_query_rewrite():
    pm = PromptManager()
    prompt = pm.query_rewrite("tell me about rna")
    assert "USER QUERY: tell me about rna" in prompt

def test_claim_extraction():
    pm = PromptManager()
    prompt = pm.claim_extraction("The sky is blue. Water is wet.")
    assert "TEXT: The sky is blue. Water is wet." in prompt
    assert "valid JSON" in prompt

def test_judge_prompt():
    pm = PromptManager()
    claims = ["Sky is blue"]
    results = [{"claim": "Sky is blue", "evidence": "Rayleigh scattering", "supported": "Yes"}]
    prompt = pm.judge_prompt(claims, results)
    
    assert "CLAIMS MADE:\n- Sky is blue" in prompt
    assert "Evidence: Rayleigh scattering" in prompt
    assert "Supported: Yes" in prompt
    assert "FAITHFULNESS_SCORE:" in prompt
    assert "REPAIR_NEEDED:" in prompt
