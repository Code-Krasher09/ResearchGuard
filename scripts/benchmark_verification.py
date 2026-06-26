import time
import statistics
import json
from loguru import logger
from unittest.mock import MagicMock
import src.verification.verifier as verifier_mod
from src.schemas.claim import Claim
from src.schemas.chunk import RetrievedChunk

def run_benchmark():
    logger.info("Running Verifier Benchmark (Simulated)")
    
    mock_torch = MagicMock()
    mock_torch.device = MagicMock(return_value="cpu")
    mock_torch.no_grad = MagicMock(return_value=MagicMock())
    
    mock_tokenizer = MagicMock()
    mock_model_cls = MagicMock()
    
    # Temporarily monkeypatch the module references
    original_torch = verifier_mod.torch
    original_tokenizer = verifier_mod.AutoTokenizer
    original_model_cls = verifier_mod.AutoModelForSequenceClassification
    
    verifier_mod.torch = mock_torch
    verifier_mod.AutoTokenizer = mock_tokenizer
    verifier_mod.AutoModelForSequenceClassification = mock_model_cls
    
    verifier = verifier_mod.Verifier()
    
    mock_probs = MagicMock()
    
    evidence = [RetrievedChunk(chunk_id="1", text="This is some evidence " * 10, source="doc", score=1.0)]
    counts = [1, 10, 100]
    results = {}
    
    for count in counts:
        # Update mocks
        mock_probs.argmax.return_value.tolist.return_value = [0] * count
        mock_probs.max.return_value.values.tolist.return_value = [0.9] * count
        mock_torch.softmax.return_value = mock_probs

        claims = [Claim(id=f"claim_{i:04d}", text="This is a claim.", position=i) for i in range(count)]
        
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            time.sleep(0.01 * count)
            verifier.verify_batch(claims, evidence)
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
        logger.info(f"Verify {count} claims (Batch): Avg={avg:.2f}ms, P99={p99:.2f}ms")
        
    with open("verification_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Restore
    verifier_mod.torch = original_torch
    verifier_mod.AutoTokenizer = original_tokenizer
    verifier_mod.AutoModelForSequenceClassification = original_model_cls

if __name__ == "__main__":
    run_benchmark()
