import time
import statistics
import json
from loguru import logger
from src.generation.prompts import PromptManager

def run_benchmark():
    logger.info("Running Prompt Rendering Benchmark")
    
    counts = [100, 1000, 10000]
    results = {}
    
    context = "X is Y. " * 50
    query = "What is X?"
    
    for count in counts:
        latencies = []
        for _ in range(count):
            start = time.perf_counter()
            PromptManager.scientific_qa(query, context)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
        avg = statistics.mean(latencies)
        p99 = sorted(latencies)[int(count * 0.99) - 1]
        
        results[str(count)] = {
            "avg_ms": round(avg, 4),
            "p99_ms": round(p99, 4)
        }
        logger.info(f"Render {count} prompts: Avg={avg:.4f}ms, P99={p99:.4f}ms")
        
    with open("prompt_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_benchmark()
