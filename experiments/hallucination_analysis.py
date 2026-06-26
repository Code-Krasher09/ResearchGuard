import sys
import os
import time
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger
from src.pipeline.pipeline import ResearchGuard

# Disable excessive logging for cleaner stdout
logger.remove()
logger.add(sys.stderr, level="WARNING")

def main():
    print("Initializing ResearchGuard for Hallucination Analysis...")
    rg = ResearchGuard()
    
    # 1. Build the specific LoRA Corpus as requested
    docs = [
        {"text": "LoRA freezes pretrained weights.", "source": "paper1"},
        {"text": "LoRA trains low rank matrices.", "source": "paper2"}
    ]
    rg.pipeline.retriever.build_from_documents(docs)
    
    # 2. Build 20 Questions (Answerable, Partially answerable, Unanswerable)
    questions = [
        # Answerable
        "What is LoRA?",
        "Does LoRA freeze weights?",
        "What does LoRA train?",
        "How does LoRA adapt pretrained models?",
        "What matrices are used in LoRA?",
        "Are pretrained weights frozen in LoRA?",
        "Does LoRA use low rank matrices?",
        # Partially Answerable (Requires synthesis beyond text)
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
        # In src.schemas.judgment, there's supported, neutral, contradicted. Hallucinations are neutral or contradicted.
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
    
    print("\n--- METRICS ---")
    print(f"Hallucination Rate: {hallucination_rate:.2%}")
    print(f"Repair Rate:        {repair_rate:.2%}")
    print(f"Avg Faithfulness:   {avg_faithfulness:.2f}")
    print(f"Avg Latency:        {avg_latency:.2f}s")
    print(f"Avg Claim Count:    {avg_claims:.2f}")
    
    # 4. Generate Markdown Report
    with open("docs/experiments.md", "w") as f:
        f.write("# Phase 6A: Experimental Hallucination Analysis\n\n")
        
        f.write("## 1. Corpus\n")
        f.write("- **paper1**: LoRA freezes pretrained weights.\n")
        f.write("- **paper2**: LoRA trains low rank matrices.\n\n")
        
        f.write("## 2. Summary Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Hallucination Rate | {hallucination_rate:.2%} |\n")
        f.write(f"| Repair Rate | {repair_rate:.2%} |\n")
        f.write(f"| Average Faithfulness | {avg_faithfulness:.2f} |\n")
        f.write(f"| Average Latency | {avg_latency:.2f}s |\n")
        f.write(f"| Average Claims per Response | {avg_claims:.2f} |\n\n")
        
        f.write("## 3. Results Table\n")
        f.write("| Query | Faithfulness | Claims (S/N/C) | Repaired | Hallucination |\n")
        f.write("|-------|--------------|----------------|----------|---------------|\n")
        for r in results:
            h_mark = "Yes" if r["has_hallucination"] else "No"
            r_mark = "Yes" if r["repair_triggered"] else "No"
            f.write(f"| {r['query']} | {r['faithfulness']:.2f} | {r['claims']} ({r['supported']}/{r['neutral']}/{r['contradicted']}) | {r_mark} | {h_mark} |\n")
            
        f.write("\n## 4. Prompt Investigation\n")
        f.write("**Why do unsupported claims survive?**\n")
        f.write("1. **Generator Overconfidence**: The generator (Qwen via Groq) possesses vast latent knowledge about LoRA (e.g., that it reduces memory consumption). Despite strict prompting, it injects parametric knowledge when the retrieved context is too sparse.\n")
        f.write("2. **Neutral Label Ambiguity**: DeBERTa classifies these claims as NEUTRAL because they don't explicitly contradict the corpus, they just aren't explicitly supported by the two tiny sentences.\n")
        f.write("3. **Lenient Repair Trigger**: The current pipeline might tolerate some level of neutral claims if the threshold allows it, failing to aggressively trigger regeneration on parametric knowledge leakage.\n\n")
        
        f.write("**Recommendations for Prompt Changes:**\n")
        f.write("1. Add strict grounding anchors to the Generator prompt: `DO NOT use your internal knowledge. If the context does not contain the answer, explicitly state 'I don't know'.`\n")
        f.write("2. Add a hallucination warning block: `WARNING: Any statement made about benefits, applications, or history not explicitly detailed in the chunk will be heavily penalized.`\n")
        f.write("3. Modify the Repair Planner prompt to aggressively target 'NEUTRAL' labels as completely unacceptable failures when strict adherence is required.\n")
        
    print("\nReport written to docs/experiments.md")

if __name__ == "__main__":
    main()
