import time
import statistics
import json
from loguru import logger
from src.repair.planner import RepairPlanner
from src.schemas.query import Query
from src.schemas.judgment import Judgment
from src.schemas.verification import VerificationResult

def run_benchmark():
    logger.info("Running Planner Benchmark")
    planner = RepairPlanner()
    counts = [1, 10, 100, 1000]
    results = {}
    
    query = Query.create(text="What is X?")
    judgment = Judgment(faithfulness_score=0.0, supported=0, neutral=0, contradicted=1, total_claims=1, repair_needed=True, reason="Contradiction found", confidence=0.9)
    res = [VerificationResult(claim_id="1", claim="c", evidence_chunk_ids=["1"], top_evidence="e", label="CONTRADICTED", top_confidence=0.9)]
    
    for count in counts:
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            for _ in range(count):
                planner.plan(query, judgment, res)
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
        logger.info(f"Plan {count} times: Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("planner_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
