import pytest
from unittest.mock import MagicMock, patch
from src.pipeline.pipeline import ResearchGuard
from src.schemas.pipeline import PipelineComponents
from src.schemas.pipeline_result import PipelineResult

from src.schemas.judgment import Judgment
from src.schemas.repair_result import RepairResult
from src.schemas.evaluation import EvaluationResult
from src.schemas.repair import RepairStrategy

@pytest.fixture
def mock_pipeline():
    return PipelineComponents(
        retriever=MagicMock(),
        generator=MagicMock(),
        claim_extractor=MagicMock(),
        verifier=MagicMock(),
        judge=MagicMock(),
        planner=MagicMock()
    )

def test_pipeline_success(mock_pipeline):
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(faithfulness=1.0, context_precision=1.0, context_recall=1.0, answer_relevancy=1.0, repair_rate=0.0)
    rg = ResearchGuard(pipeline=mock_pipeline, evaluator=evaluator)
    
    with patch.object(rg, 'executor') as mock_executor:
        mock_judgment = Judgment(total_claims=1, supported=1, neutral=0, contradicted=0, faithfulness_score=1.0, repair_needed=False, reason="", confidence=1.0)
        mock_result = RepairResult(success=True, answer="Answer", judgment=mock_judgment, attempt=1, strategy=RepairStrategy.NONE, latency=0.0)
        mock_executor.execute.return_value = mock_result
        
        result = rg.run("Query?")
        
        assert isinstance(result, PipelineResult)
        assert result.answer == "Answer"
        assert result.repair_result.attempt == 1
        evaluator.evaluate.assert_called_once()

def test_pipeline_repair_triggered(mock_pipeline):
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(faithfulness=1.0, context_precision=1.0, context_recall=1.0, answer_relevancy=1.0, repair_rate=1.0)
    rg = ResearchGuard(pipeline=mock_pipeline, evaluator=evaluator)
    
    with patch.object(rg, 'executor') as mock_executor:
        mock_judgment = Judgment(total_claims=1, supported=1, neutral=0, contradicted=0, faithfulness_score=1.0, repair_needed=False, reason="", confidence=1.0)
        mock_result = RepairResult(success=True, answer="Repaired Answer", judgment=mock_judgment, attempt=2, strategy=RepairStrategy.INCREASE_K, latency=0.0)
        mock_executor.execute.return_value = mock_result
        
        result = rg.run("Query?")
        
        assert result.repair_result.attempt == 2
        assert result.answer == "Repaired Answer"
        evaluator.evaluate.assert_called_once()

def test_pipeline_failure(mock_pipeline):
    evaluator = MagicMock()
    evaluator.evaluate.return_value = EvaluationResult(faithfulness=0.0, context_precision=1.0, context_recall=1.0, answer_relevancy=1.0, repair_rate=1.0)
    rg = ResearchGuard(pipeline=mock_pipeline, evaluator=evaluator)
    
    with patch.object(rg, 'executor') as mock_executor:
        mock_judgment = Judgment(total_claims=1, supported=0, neutral=0, contradicted=1, faithfulness_score=0.0, repair_needed=True, reason="Contradiction", confidence=1.0)
        mock_result = RepairResult(success=False, answer="Bad Answer", judgment=mock_judgment, attempt=3, strategy=RepairStrategy.QUERY_REWRITE, latency=0.0)
        mock_executor.execute.return_value = mock_result
        
        result = rg.run("Query?")
        
        assert result.repair_result.attempt == 3
        assert result.repair_result.success is False
        evaluator.evaluate.assert_called_once()
