import sys
import os
import time
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger
from src.pipeline.pipeline import ResearchGuard

# Disable excessive logging for cleaner stdout
logger.remove()
logger.add(sys.stderr, level="WARNING")

def main():
    print("Initializing ResearchGuard for Hallucination Analysis on LoRA Corpus...")
    rg = ResearchGuard()
    
    # 1. Build the specific LoRA Corpus
    pkl_path = "data/processed/lora_chunks.pkl"
    with open(pkl_path, "rb") as f:
        chunks = pickle.load(f)
    rg.pipeline.retriever.build_from_chunks(chunks)
    
    # 2. Build 20 Questions
    questions = [
        # Answerable
        "What is LoRA?",
        "Does LoRA freeze weights?",
        "What does LoRA train?",
        "How does LoRA adapt pretrained models?",
        "What matrices are used in LoRA?",
        "Are pretrained weights frozen in LoRA?",
        "Does LoRA use low rank matrices?",
        # Partially Answerable
        "Why does LoRA freeze weights?",
        "Does LoRA reduce memory consumption?",
        "Does LoRA accelerate training?",
        "Is LoRA better than full fine-tuning?",
        "What are the benefits of low rank matrices in LoRA?",
        "How is LoRA applied to large language models?",
        "Does LoRA improve inference latency?",
        # Unanswerable
        "Who invented LoRA?",
        "When was LoRA published?",
        "What disease causes asthma?",
        "How do you cook spaghetti?",
        "What is the capital of France?",
        "Who won the 1994 World Cup?"
    ]
    
    results = []
    
    print(f"\nRunning analysis on {len(questions)} queries...\n")
    
    start_all = time.perf_counter()
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Query: {q}")
        res = rg.run(q)
        
        # Analyze unsupported claims
        unsupported = res.judgment.neutral + res.judgment.contradicted
        has_hallucination = unsupported > 0
        
        results.append({
            "query": q,
            "answer": res.answer,
            "latency": res.latency,
            "claims": res.judgment.total_claims,
            "supported": res.judgment.supported,
            "neutral": res.judgment.neutral,
            "contradicted": res.judgment.contradicted,
            "faithfulness": res.judgment.faithfulness_score,
            "repair_triggered": res.repair_result.attempt > 1,
            "has_hallucination": has_hallucination
        })
    
    total_time = time.perf_counter() - start_all
    
    # 3. Calculate Metrics
    total_queries = len(questions)
    hallucination_rate = sum(1 for r in results if r["has_hallucination"]) / total_queries
    repair_rate = sum(1 for r in results if r["repair_triggered"]) / total_queries
    avg_faithfulness = sum(r["faithfulness"] for r in results) / total_queries
    avg_latency = total_time / total_queries
    avg_claims = sum(r["claims"] for r in results) / total_queries
    
    print("\n--- METRICS (LoRA Corpus) ---")
    print(f"Hallucination Rate: {hallucination_rate:.2%}")
    print(f"Repair Rate:        {repair_rate:.2%}")
    print(f"Avg Faithfulness:   {avg_faithfulness:.2f}")
    print(f"Avg Latency:        {avg_latency:.2f}s")
    print(f"Avg Claim Count:    {avg_claims:.2f}")
    
    # Existing synthetic metrics from Phase 6A journal
    synth_hallucination = 0.95
    synth_repair = 0.95
    synth_faithfulness = 0.28
    synth_latency = 4.45
    synth_claims = 2.35
    
    # 4. Generate Markdown Report
    with open("docs/experiments_lora.md", "w") as f:
        f.write("# Phase 6C-3: Experimental Hallucination Analysis on Rich Corpus\n\n")
        f.write("## 1. Corpus Comparison\n")
        f.write("| Metric | Synthetic Corpus (Phase 6A) | LoRA Corpus (Phase 6C-3) | Delta |\n")
        f.write("|--------|-----------------------------|--------------------------|-------|\n")
        
        h_delta = hallucination_rate - synth_hallucination
        r_delta = repair_rate - synth_repair
        f_delta = avg_faithfulness - synth_faithfulness
        l_delta = avg_latency - synth_latency
        c_delta = avg_claims - synth_claims
        
        f.write(f"| Hallucination Rate | {synth_hallucination:.2%} | {hallucination_rate:.2%} | {h_delta:+.2%} |\n")
        f.write(f"| Repair Rate | {synth_repair:.2%} | {repair_rate:.2%} | {r_delta:+.2%} |\n")
        f.write(f"| Avg Faithfulness | {synth_faithfulness:.2f} | {avg_faithfulness:.2f} | {f_delta:+.2f} |\n")
        f.write(f"| Avg Latency | {synth_latency:.2f}s | {avg_latency:.2f}s | {l_delta:+.2f}s |\n")
        f.write(f"| Avg Claims | {synth_claims:.2f} | {avg_claims:.2f} | {c_delta:+.2f} |\n\n")
        
        f.write("## 2. Results Table\n")
        f.write("| Query | Faithfulness | Claims (S/N/C) | Repaired | Hallucination |\n")
        f.write("|-------|--------------|----------------|----------|---------------|\n")
        for r in results:
            h_mark = "Yes" if r["has_hallucination"] else "No"
            r_mark = "Yes" if r["repair_triggered"] else "No"
            f.write(f"| {r['query']} | {r['faithfulness']:.2f} | {r['claims']} ({r['supported']}/{r['neutral']}/{r['contradicted']}) | {r_mark} | {h_mark} |\n")
            
    print("\nReport written to docs/experiments_lora.md")

if __name__ == "__main__":
    main()
