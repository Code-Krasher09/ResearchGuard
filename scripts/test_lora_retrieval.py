import time
import pickle
from src.retrieval.retriever import Retriever

QUERIES = [
    "What is LoRA?",
    "Who proposed LoRA?",
    "How does LoRA reduce memory?",
    "What matrices are trained?",
    "What is rank r?",
    "What weights remain frozen?",
    "What are the advantages of LoRA?",
    "How many trainable parameters are reduced?",
    "What datasets were used?",
    "What experiments were conducted?"
]

# Keywords that should be present in a good Top 5 result for each query to calculate an approximate Recall@5
GROUND_TRUTH_KEYWORDS = {
    "What is LoRA?": ["low-rank adaptation", "lora"],
    "Who proposed LoRA?": ["edward hu", "yelong shen", "microsoft"],
    "How does LoRA reduce memory?": ["memory", "gpu", "hardware", "reduce"],
    "What matrices are trained?": ["trainable rank decomposition matrices", "update matrices", "b", "a"],
    "What is rank r?": ["rank r", "rank-deficient", "low-rank"],
    "What weights remain frozen?": ["freezes the pre-trained model weights", "frozen"],
    "What are the advantages of LoRA?": ["inference latency", "throughput", "memory"],
    "How many trainable parameters are reduced?": ["10,000 times", "parameters", "reduce"],
    "What datasets were used?": ["wikisql", "mnli", "samsum", "e2e"],
    "What experiments were conducted?": ["gpt-2", "gpt-3", "roberta", "deberta"]
}

def safe_print(text):
    print(str(text).encode('ascii', 'replace').decode('ascii'))

def run_retrieval_validation():
    pkl_path = "data/processed/lora_chunks.pkl"
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
        
    safe_print(f"Loaded {len(chunks)} chunks.")
    
    retriever = Retriever()
    retriever.build_from_chunks(chunks)
    
    total_latency = 0.0
    total_score = 0.0
    recall_at_5_hits = 0
    
    for i, q in enumerate(QUERIES, 1):
        safe_print("=" * 80)
        safe_print(f"Q{i}: {q}")
        safe_print("=" * 80)
        
        start_time = time.perf_counter()
        results = retriever.retrieve(q, k=5)
        latency = (time.perf_counter() - start_time) * 1000
        total_latency += latency
        
        # Recall@5 check
        keywords = GROUND_TRUTH_KEYWORDS.get(q, [])
        hit = False
        
        for idx, res in enumerate(results, 1):
            if idx == 1:
                total_score += res.score
            
            text_lower = res.text.lower()
            if any(k.lower() in text_lower for k in keywords):
                hit = True
                
            safe_print(f"Rank {idx} | Score: {res.score:.4f} | Section: {res.section} | Page: {res.page}")
            safe_print(f"Preview: {res.text[:150]}...\n")
            
        if hit:
            recall_at_5_hits += 1
            
        safe_print(f"[Latency: {latency:.2f}ms] [Recall Hit: {hit}]")
        
    avg_latency = total_latency / len(QUERIES)
    avg_score = total_score / len(QUERIES)
    recall_at_5 = recall_at_5_hits / len(QUERIES)
    
    safe_print("\n" + "=" * 80)
    safe_print("RETRIEVAL METRICS")
    safe_print("=" * 80)
    safe_print(f"Average Retrieval Latency: {avg_latency:.2f} ms")
    safe_print(f"Average Top-1 Score: {avg_score:.4f}")
    safe_print(f"Recall@5 (Heuristic): {recall_at_5:.2f}")

if __name__ == "__main__":
    run_retrieval_validation()
