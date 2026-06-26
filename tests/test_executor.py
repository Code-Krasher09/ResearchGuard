import pytest
from unittest.mock import MagicMock
from src.schemas.query import Query
from src.schemas.repair import RepairPlan, RepairStrategy
from src.schemas.judgment import Judgment
from src.schemas.claim import Claim
from src.schemas.verification import VerificationResult
from src.schemas.pipeline import PipelineComponents
from src.repair.executor import RepairExecutor, MAX_REPAIR_ATTEMPTS

@pytest.fixture
def executor():
    return RepairExecutor()

@pytest.fixture
def pipeline():
    return PipelineComponents(
        retriever=MagicMock(),
        generator=MagicMock(),
        claim_extractor=MagicMock(),
        verifier=MagicMock(),
        judge=MagicMock(),
        planner=MagicMock()
    )

def test_execute_success_first_attempt(executor, pipeline):
    query = Query.create(text="What is X?")
    plan = RepairPlan(strategy=RepairStrategy.INCREASE_K, reason="test", current_k=5, new_k=10, confidence=0.9)
    
    pipeline.retriever.retrieve.return_value = []
    
    # Mock GeneratedAnswer string return
    mock_answer = MagicMock()
    mock_answer.answer = "Answer"
    pipeline.generator.generate_answer.return_value = mock_answer
    
    pipeline.claim_extractor.extract_claims.return_value = []
    pipeline.verifier.verify_batch.return_value = []
    
    good_judgment = Judgment(faithfulness_score=1.0, supported=1, neutral=0, contradicted=0, total_claims=1, repair_needed=False, reason="Faithful", confidence=1.0)
    pipeline.judge.judge.return_value = good_judgment
    
    result = executor.execute(query, plan, pipeline)
    
    assert result.success is True
    assert result.attempt == 1
    assert result.judgment == good_judgment
    assert len(result.repair_history) == 1
    pipeline.planner.plan.assert_not_called()

def test_execute_max_attempts(executor, pipeline):
    query = Query.create(text="What is X?")
    plan = RepairPlan(strategy=RepairStrategy.QUERY_REWRITE, reason="test", current_k=5, rewrite_required=True, confidence=0.9)
    
    mock_answer = MagicMock()
    mock_answer.answer = "Answer"
    pipeline.generator.generate_answer.return_value = mock_answer
    
    bad_judgment = Judgment(faithfulness_score=0.0, supported=0, neutral=0, contradicted=1, total_claims=1, repair_needed=True, reason="Contradicted", confidence=0.9)
    pipeline.judge.judge.return_value = bad_judgment
    
    pipeline.planner.plan.return_value = RepairPlan(strategy=RepairStrategy.INCREASE_K, reason="test", current_k=5, new_k=10, confidence=0.9)
    
    result = executor.execute(query, plan, pipeline)
    
    assert result.success is False
    assert result.attempt == MAX_REPAIR_ATTEMPTS
    assert pipeline.judge.judge.call_count == MAX_REPAIR_ATTEMPTS
    assert pipeline.planner.plan.call_count == MAX_REPAIR_ATTEMPTS - 1
