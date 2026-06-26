import time
import statistics
import json
from loguru import logger
from src.evaluation.evaluator import Evaluator

def run_benchmark():
    logger.info("Running Evaluator Benchmark")
    evaluator = Evaluator()
    counts = [1, 10, 100]
    results = {}
    
    for count in counts:
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            for _ in range(count):
                evaluator.evaluate("What is X?", "X is Y.", ["X is Y"], "Y")
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
        logger.info(f"Evaluate {count} times: Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("evaluator_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
