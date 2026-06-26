import time
import statistics
import json
from loguru import logger
from src.verification.claims import ClaimExtractor

def run_benchmark():
    logger.info("Running Claim Extraction Benchmark")
    try:
        extractor = ClaimExtractor()
    except RuntimeError as e:
        logger.error(f"Cannot run benchmark: {e}")
        return
        
    answer = "LoRA freezes pretrained weights. LoRA trains low rank matrices. " * 5
    counts = [100, 1000, 10000]
    results = {}
    
    for count in counts:
        latencies = []
        for _ in range(count):
            start = time.perf_counter()
            extractor.extract_claims(answer)
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
        logger.info(f"Extract {count} answers: Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("claim_extraction_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
