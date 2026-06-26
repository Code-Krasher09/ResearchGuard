import time
import json
import statistics
import os
from loguru import logger

# Try to import client, but gracefully mock if not properly configured with an API key
try:
    from src.generation.client import GroqClient
    client = GroqClient()
    HAS_CLIENT = True
except ValueError as e:
    logger.warning(f"Groq API Key not found. Benchmarks will run in simulated mode. Error: {e}")
    HAS_CLIENT = False

def run_benchmark():
    logger.info("Running Groq Generation Benchmark")
    
    prompt = "Reply with exactly one word: Pong."
    counts = [1, 10, 50, 100]
    
    results = {}
    
    for count in counts:
        logger.info(f"Running {count} iterations...")
        latencies = []
        for i in range(count):
            start = time.perf_counter()
            if HAS_CLIENT:
                try:
                    client.generate(prompt, max_tokens=2)
                except Exception as e:
                    logger.error(f"Failed at iteration {i}: {e}")
            else:
                # Simulate latency for testing environments without a real key
                time.sleep(0.01) # 10ms simulated latency
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # ms
            
        if latencies:
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
            logger.info(f"Results for {count} iterations: {results[str(count)]}")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved results to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()
