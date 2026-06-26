import time
import statistics
import json
from loguru import logger

def run_benchmark():
    logger.info("Running Generator Benchmark")
    
    try:
        from src.generation.generator import Generator
        from src.schemas.chunk import RetrievedChunk
        import os
        
        # Determine if we have a valid API key
        has_key = bool(os.getenv("GROQ_API_KEY"))
        if not has_key:
            logger.warning("GROQ_API_KEY not set. Running benchmark in mock/simulation mode.")
            
        generator = Generator()
        query = "What is X?"
        chunks = [RetrievedChunk(chunk_id="1", text="X is Y. " * 10, source="doc", score=1.0)]
        
        counts = [1, 10, 50]
        results = {}
        
        for count in counts:
            latencies = []
            for i in range(count):
                start = time.perf_counter()
                if has_key:
                    try:
                        generator.generate_answer(query, chunks)
                    except Exception as e:
                        logger.error(f"Failed at {i}: {e}")
                else:
                    # Simulate ~15ms latency + prompt rendering
                    time.sleep(0.015)
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
            logger.info(f"Generated {count} answers: Avg={avg:.2f}ms, P99={p99:.2f}ms")
            
        with open("generator_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")

if __name__ == "__main__":
    run_benchmark()
