import time
from loguru import logger
from src.retrieval.retriever import Retriever

def main():
    logger.info("Initializing Retrieval Smoke Test")
    
    docs = [
        {"text":"LoRA freezes pretrained weights.", "source":"paper1"},
        {"text":"LoRA trains low rank matrices.", "source":"paper2"}
    ]
    
    start_time = time.perf_counter()
    
    retriever = Retriever()
    retriever.build_from_documents(docs)
    
    query = "What is LoRA?"
    
    query_start = time.perf_counter()
    results = retriever.retrieve(query, k=2)
    query_latency = time.perf_counter() - query_start
    
    total_latency = time.perf_counter() - start_time
    
    print("\n--- RETRIEVAL RESULTS ---")
    print(f"Query: {query}")
    for res in results:
        print(f"Score: {res.score:.4f} | Source: {res.source} | Text: {res.text}")
    print("-------------------------\n")
    
    logger.info(f"Single Query Latency: {query_latency * 1000:.2f} ms")
    logger.info(f"Total Latency (Init+Build+Query): {total_latency:.2f} s")

if __name__ == "__main__":
    main()
