import numpy as np
from typing import List, Optional
from loguru import logger
from src.schemas.evaluation import EvaluationResult

try:
    import ragas
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate as ragas_evaluate
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    ragas_evaluate = None
    faithfulness = None
    answer_relevancy = None
    context_precision = None
    context_recall = None
    Dataset = None
    logger.warning("RAGAS not installed. Using simple fallback heuristics for evaluation.")

class Evaluator:
    """
    Evaluates pipeline quality using RAGAS if available, or simple heuristics otherwise.
    """
    def __init__(self, embedder=None):
        self.embedder = embedder
        self.total_samples = 0
        self.total_repairs = 0
        
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        if not self.embedder:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2: return 0.0
            return len(words1.intersection(words2)) / len(words1.union(words2))
            
        v1 = self.embedder.embed_query(text1)
        v2 = self.embedder.embed_query(text2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        
    def evaluate(self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None, repair_triggered: bool = False) -> EvaluationResult:
        self.total_samples += 1
        if repair_triggered:
            self.total_repairs += 1
        repair_rate = self.total_repairs / self.total_samples
        
        if RAGAS_AVAILABLE:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]
            if ground_truth:
                metrics.append(context_recall)
                
            try:
                result = ragas_evaluate(dataset, metrics=metrics)
                return EvaluationResult(
                    faithfulness=result.get("faithfulness", 0.0),
                    context_precision=result.get("context_precision", 0.0),
                    context_recall=result.get("context_recall", 0.0),
                    answer_relevancy=result.get("answer_relevancy", 0.0),
                    repair_rate=repair_rate
                )
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}. Falling back to heuristics.")
        
        # Fallback Heuristics with Semantic Similarity
        faithfulness_score = max([self._cosine_similarity(answer, c) for c in contexts]) if contexts else 0.0
        relevancy_score = self._cosine_similarity(answer, question)
        precision_score = 1.0 if len(contexts) > 0 else 0.0
        recall_score = self._cosine_similarity(answer, ground_truth) if ground_truth else 0.0
        
        return EvaluationResult(
            faithfulness=min(1.0, max(0.0, faithfulness_score)),
            context_precision=precision_score,
            context_recall=min(1.0, max(0.0, recall_score)),
            answer_relevancy=min(1.0, max(0.0, relevancy_score)),
            repair_rate=repair_rate
        )
