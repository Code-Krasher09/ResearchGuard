import time
import statistics
import json
from loguru import logger
from src.verification.judge import Judge
from src.schemas.claim import Claim
from src.schemas.verification import VerificationResult

def run_benchmark():
    logger.info("Running Judge Benchmark")
    judge = Judge()
    counts = [1, 10, 100, 1000]
    results = {}
    
    for count in counts:
        claims = [Claim(id=str(i), text="claim", position=i) for i in range(count)]
        verifications = [
            VerificationResult(claim_id=str(i), claim="claim", evidence_chunk_ids=[], top_evidence="", label="SUPPORTED", top_confidence=0.9)
            for i in range(count)
        ]
        
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            judge.judge(claims, verifications)
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
        logger.info(f"Judge {count} claims: Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("judge_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
