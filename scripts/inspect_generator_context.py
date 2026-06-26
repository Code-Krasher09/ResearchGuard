import pickle
from src.retrieval.retriever import Retriever

QUERIES = [
    "What is LoRA?",
    "Who proposed LoRA?",
    "How does LoRA reduce memory?"
]

def safe_print(text):
    print(str(text).encode('ascii', 'replace').decode('ascii'))

def inspect_context():
    pkl_path = "data/processed/lora_chunks.pkl"
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
        
    retriever = Retriever()
    retriever.build_from_chunks(chunks)
    
    token_counts = []
    
    safe_print("=" * 60)
    safe_print("GENERATOR CONTEXT INSPECTION")
    safe_print("=" * 60)
    
    for i, q in enumerate(QUERIES, 1):
        safe_print(f"\nQUERY {i}: {q}")
        safe_print("-" * 60)
        
        results = retriever.retrieve(q, k=5)
        
        context_text = ""
        for idx, res in enumerate(results, 1):
            safe_print(f"Chunk {idx} | Score: {res.score:.4f} | Section: {res.section} | Page: {res.page}")
            safe_print(f"Text: {res.text}\n")
            context_text += res.text + "\n"
            
        # Estimate tokens using the standard len // 4 heuristic used across ResearchGuard
        total_tokens = len(context_text) // 4
        token_counts.append(total_tokens)
        
        safe_print(f">>> Total Estimated Context Tokens: {total_tokens} <<<\n")
        
    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    min_tokens = min(token_counts)
    
    safe_print("=" * 60)
    safe_print("CONTEXT METRICS")
    safe_print("=" * 60)
    safe_print(f"Average context tokens: {avg_tokens:.1f}")
    safe_print(f"Max context tokens: {max_tokens}")
    safe_print(f"Min context tokens: {min_tokens}")

if __name__ == "__main__":
    inspect_context()
