import time
from loguru import logger
from src.schemas.claim import Claim
from src.schemas.chunk import RetrievedChunk
from src.verification.verifier import Verifier

def main():
    logger.info("Initializing Verifier Smoke Test")
    
    start_time = time.perf_counter()
    verifier = Verifier()
    
    claim = Claim(id="1", text="LoRA freezes pretrained weights.", position=1)
    chunk = RetrievedChunk(
        chunk_id="1",
        text="LoRA freezes pretrained weights.",
        source="paper1",
        score=1.0
    )
    
    query_start = time.perf_counter()
    result = verifier.verify_claim(claim, [chunk])
    query_latency = time.perf_counter() - query_start
    
    total_latency = time.perf_counter() - start_time
    
    print("\n--- VERIFICATION RESULTS ---")
    print(f"Claim: {result.claim}")
    print(f"Label: {result.label} | Confidence: {result.top_confidence:.4f}")
    print("----------------------------\n")
    
    logger.info(f"Single Claim Verification Latency: {query_latency * 1000:.2f} ms")
    logger.info(f"Total Latency (Init+Query): {total_latency:.2f} s")

if __name__ == "__main__":
    main()
