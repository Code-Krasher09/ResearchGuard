import time
import statistics
import json
from loguru import logger
from unittest.mock import MagicMock, patch
from src.pipeline.pipeline import ResearchGuard
from src.schemas.pipeline import PipelineComponents

def run_benchmark():
    logger.info("Running Pipeline Benchmark")
    
    pipeline = PipelineComponents(
        retriever=MagicMock(),
        generator=MagicMock(),
        claim_extractor=MagicMock(),
        verifier=MagicMock(),
        judge=MagicMock(),
        planner=MagicMock()
    )
    evaluator = MagicMock()
    rg = ResearchGuard(pipeline=pipeline, evaluator=evaluator)
    
    counts = [1, 10, 100]
    results = {}
    
    with patch.object(rg, 'executor') as mock_executor:
        from src.schemas.judgment import Judgment
        from src.schemas.repair_result import RepairResult
        from src.schemas.evaluation import EvaluationResult
        
        from src.schemas.repair import RepairStrategy
        mock_judgment = Judgment(total_claims=1, supported=1, neutral=0, contradicted=0, faithfulness_score=1.0, repair_needed=False, reason="", confidence=1.0)
        mock_result = RepairResult(success=True, answer="Answer", judgment=mock_judgment, attempt=1, strategy=RepairStrategy.NONE, latency=0.0)
        evaluator.evaluate.return_value = EvaluationResult(faithfulness=1.0, context_precision=1.0, context_recall=1.0, answer_relevancy=1.0, repair_rate=0.0)
        mock_executor.execute.return_value = mock_result
        
        for count in counts:
            latencies = []
            for _ in range(10):
                start = time.perf_counter()
                for _ in range(count):
                    rg.run("What is X?")
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
                
            avg = statistics.mean(latencies)
            latencies.sort()
            p50 = latencies[int(len(latencies)*0.5)]
            p95 = latencies[int(len(latencies)*0.95)]
            p99 = latencies[int(len(latencies)*0.99)]
            
            results[str(count)] = {
                "avg_ms": round(avg, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2)
            }
            logger.info(f"Execute {count} queries: Avg={avg:.2f}ms, P99={p99:.2f}ms")
            
    with open("pipeline_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
