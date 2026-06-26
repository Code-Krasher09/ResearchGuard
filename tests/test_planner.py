import pytest
from src.schemas.query import Query
from src.schemas.judgment import Judgment
from src.schemas.verification import VerificationResult
from src.repair.planner import RepairPlanner

@pytest.fixture
def planner():
    return RepairPlanner()

@pytest.fixture
def dummy_query():
    return Query.create(text="What is X?")

def test_planner_no_repair(planner, dummy_query):
    judgment = Judgment(faithfulness_score=1.0, supported=1, neutral=0, contradicted=0, total_claims=1, repair_needed=False, reason="Faithful", confidence=0.9)
    plan = planner.plan(dummy_query, judgment, [])
    assert plan.strategy == "NONE"

def test_planner_empty_evidence(planner, dummy_query):
    judgment = Judgment(faithfulness_score=0.0, supported=0, neutral=0, contradicted=0, total_claims=0, repair_needed=True, reason="Empty claims or verification results", confidence=0.0)
    plan = planner.plan(dummy_query, judgment, [])
    assert plan.strategy == "HYBRID"
    assert plan.new_k == 10  # max(5, 5*2) = 10
    assert plan.rewrite_required

def test_planner_contradiction(planner, dummy_query):
    judgment = Judgment(faithfulness_score=0.0, supported=0, neutral=0, contradicted=1, total_claims=1, repair_needed=True, reason="Contradiction found", confidence=0.9)
    res = [VerificationResult(claim_id="1", claim="c", evidence_chunk_ids=["1"], top_evidence="e", label="CONTRADICTED", top_confidence=0.9)]
    plan = planner.plan(dummy_query, judgment, res)
    assert plan.strategy == "QUERY_REWRITE"
    assert plan.rewrite_required

def test_planner_low_support(planner, dummy_query):
    judgment = Judgment(faithfulness_score=0.5, supported=1, neutral=1, contradicted=0, total_claims=2, repair_needed=True, reason="Low support", confidence=0.9)
    res = [
        VerificationResult(claim_id="1", claim="c", evidence_chunk_ids=["1"], top_evidence="e", label="SUPPORTED", top_confidence=0.9),
        VerificationResult(claim_id="2", claim="c2", evidence_chunk_ids=["1"], top_evidence="e", label="NEUTRAL", top_confidence=0.9)
    ]
    plan = planner.plan(dummy_query, judgment, res, current_k=10)
    assert plan.strategy == "INCREASE_K"
    assert plan.new_k == 20

def test_planner_fallback(planner, dummy_query):
    judgment = Judgment(faithfulness_score=1.0, supported=1, neutral=0, contradicted=0, total_claims=1, repair_needed=True, reason="Unknown", confidence=0.9)
    res = [VerificationResult(claim_id="1", claim="c", evidence_chunk_ids=["1"], top_evidence="e", label="SUPPORTED", top_confidence=0.9)]
    plan = planner.plan(dummy_query, judgment, res)
    assert plan.strategy == "HYBRID"
