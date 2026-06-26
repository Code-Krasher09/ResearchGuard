import time
import statistics
import json
from loguru import logger
from unittest.mock import MagicMock
from src.schemas.query import Query
from src.schemas.repair import RepairPlan, RepairStrategy
from src.schemas.judgment import Judgment
from src.schemas.pipeline import PipelineComponents
from src.repair.executor import RepairExecutor

def run_benchmark():
    logger.info("Running Executor Benchmark")
    executor = RepairExecutor()
    
    counts = [1, 10, 100]
    results = {}
    
    query = Query.create(text="What is X?")
    plan = RepairPlan(strategy=RepairStrategy.INCREASE_K, reason="test", current_k=5, new_k=10, confidence=0.9)
    
    pipeline = PipelineComponents(
        retriever=MagicMock(),
        generator=MagicMock(),
        claim_extractor=MagicMock(),
        verifier=MagicMock(),
        judge=MagicMock(),
        planner=MagicMock()
    )
    
    mock_answer = MagicMock()
    mock_answer.answer = "Answer"
    pipeline.generator.generate_answer.return_value = mock_answer
    
    good_judgment = Judgment(faithfulness_score=1.0, supported=1, neutral=0, contradicted=0, total_claims=1, repair_needed=False, reason="Faithful", confidence=1.0)
    pipeline.judge.judge.return_value = good_judgment
    
    for count in counts:
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            for _ in range(count):
                executor.execute(query, plan, pipeline)
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
        logger.info(f"Execute {count} times: Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("executor_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
