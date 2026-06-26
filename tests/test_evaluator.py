import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.evaluation.evaluator import Evaluator
from src.schemas.evaluation import EvaluationResult

import math

def test_evaluator_fallback():
    evaluator = Evaluator(embedder=MagicMock())
    evaluator.embedder.embed_query.side_effect = lambda x: np.array([1.0, 0.0]) if x == "X is Y." else np.array([1.0, 0.0]) # Perfect similarity
    
    result = evaluator.evaluate(
        question="What is X?", 
        answer="X is Y.", 
        contexts=["X is Y.", "Z is W."], 
        ground_truth="X is Y.",
        repair_triggered=True
    )
    assert isinstance(result, EvaluationResult)
    assert math.isclose(result.faithfulness, 1.0, abs_tol=1e-5)
    assert math.isclose(result.context_precision, 1.0, abs_tol=1e-5)
    assert math.isclose(result.context_recall, 1.0, abs_tol=1e-5)
    assert math.isclose(result.repair_rate, 1.0, abs_tol=1e-5)
    
@patch("src.evaluation.evaluator.Dataset")
@patch("src.evaluation.evaluator.RAGAS_AVAILABLE", True)
@patch("src.evaluation.evaluator.ragas_evaluate")
def test_evaluator_ragas(mock_evaluate, mock_dataset):
    mock_evaluate.return_value = {
        "faithfulness": 0.9,
        "context_precision": 0.8,
        "context_recall": 0.7,
        "answer_relevancy": 0.95
    }
    
    evaluator = Evaluator()
    result = evaluator.evaluate(
        question="What is X?", 
        answer="X is Y.", 
        contexts=["X is Y"], 
        ground_truth="Y",
        repair_triggered=False
    )
    assert result.faithfulness == 0.9
    assert result.context_precision == 0.8
    assert result.context_recall == 0.7
    assert result.answer_relevancy == 0.95
    assert result.repair_rate == 0.0

@patch("src.evaluation.evaluator.Dataset")
@patch("src.evaluation.evaluator.RAGAS_AVAILABLE", True)
@patch("src.evaluation.evaluator.ragas_evaluate")
def test_evaluator_ragas_exception_fallback(mock_evaluate, mock_dataset):
    mock_evaluate.side_effect = Exception("API Error")
    
    evaluator = Evaluator(embedder=MagicMock())
    evaluator.embedder.embed_query.return_value = np.array([1.0])
    
    result = evaluator.evaluate(
        question="What is X?", 
        answer="X is Y.", 
        contexts=["X is Y"], 
        ground_truth="Y"
    )
    assert math.isclose(result.faithfulness, 1.0, abs_tol=1e-5)
    assert math.isclose(result.context_precision, 1.0, abs_tol=1e-5)
