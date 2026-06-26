import sys
import os
import time
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from loguru import logger
from src.pipeline.pipeline import ResearchGuard

logger.remove()
logger.add(sys.stderr, level="WARNING")

def run_experiment(rg: ResearchGuard, questions: list, mode: str):
    results = []
    print(f"\nRunning analysis on {len(questions)} queries for mode: {mode}...\n")
    start_all = time.perf_counter()
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Query: {q}")
        
        # Override generator mode (we update ResearchGuard.run to accept kwargs, but for now we patch the generator's default temporarily)
        # Actually, we can just monkey-patch the prompt_manager temporarily
        if mode == "strict_extraction":
            original_mode = rg.pipeline.generator.prompt_manager.get_version
            rg.pipeline.generator.prompt_manager.get_version = lambda: "v5_extraction"
            def temp_gen(query, context, rewrite_required=False, m="strict_extraction"):
                return rg.pipeline.generator.__class__.generate_answer(rg.pipeline.generator, query, context, rewrite_required, mode="strict_extraction")
        else:
            def temp_gen(query, context, rewrite_required=False, m="grounded_extraction"):
                return rg.pipeline.generator.__class__.generate_answer(rg.pipeline.generator, query, context, rewrite_required, mode="grounded_extraction")
            
        original_gen = rg.pipeline.generator.generate_answer
        rg.pipeline.generator.generate_answer = temp_gen
        
        res = rg.run(q)
        
        rg.pipeline.generator.generate_answer = original_gen
        
        unsupported = res.judgment.neutral + res.judgment.contradicted
        has_hallucination = unsupported > 0
        
        # A useful answer is one that doesn't just say INSUFFICIENT EVIDENCE
        is_useful = "INSUFFICIENT EVIDENCE" not in res.answer.upper()
        
        results.append({
            "query": q,
            "answer": res.answer,
            "latency": res.latency,
            "claims": res.judgment.total_claims,
            "supported": res.judgment.supported,
            "faithfulness": res.judgment.faithfulness_score,
            "repair_triggered": res.repair_result.attempt > 1,
            "has_hallucination": has_hallucination,
            "is_useful": is_useful
        })
    
    total_time = time.perf_counter() - start_all
    return results, total_time

def main():
    print("Initializing ResearchGuard for Grounded Mode Ablation...")
    rg = ResearchGuard()
    
    pkl_path = "data/processed/lora_chunks.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            chunks = pickle.load(f)
        rg.pipeline.retriever.build_from_chunks(chunks)
    else:
        print("LoRA chunks not found. Using default text.")
    
    questions = [
        "What is LoRA?",
        "What matrices are trained?",
        "How many trainable parameters are reduced?",
        "What is rank r?",
        "What weights remain frozen?",
        "What are the advantages of LoRA?",
        "What datasets were used?",
        "What experiments were conducted?",
        "Does LoRA reduce memory?",
        "How does LoRA adapt pretrained models?"
    ]
    
    v5_results, v5_time = run_experiment(rg, questions, "strict_extraction")
    v6_results, v6_time = run_experiment(rg, questions, "grounded_extraction")
    
    def print_metrics(name, results, total_time):
        total_queries = len(questions)
        hallucination_rate = sum(1 for r in results if r["has_hallucination"]) / total_queries
        repair_rate = sum(1 for r in results if r["repair_triggered"]) / total_queries
        avg_faithfulness = sum(r["faithfulness"] for r in results) / total_queries
        avg_latency = total_time / total_queries
        useful_rate = sum(1 for r in results if r["is_useful"]) / total_queries
        
        print(f"\n--- METRICS ({name}) ---")
        print(f"Hallucination Rate: {hallucination_rate:.2%}")
        print(f"Repair Rate:        {repair_rate:.2%}")
        print(f"Avg Faithfulness:   {avg_faithfulness:.2f}")
        print(f"Useful Answer Rate: {useful_rate:.2%}")
        print(f"Avg Latency:        {avg_latency:.2f}s")
        
    print_metrics("v5_strict_extraction", v5_results, v5_time)
    print_metrics("v6_grounded_extraction", v6_results, v6_time)

if __name__ == "__main__":
    main()
