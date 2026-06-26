import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
from loguru import logger
from src.pipeline.pipeline import ResearchGuard

def main():
    print("========================================")
    print("   ResearchGuard E2E Validation Suite   ")
    print("========================================")
    
    start_init = time.perf_counter()
    rg = ResearchGuard()
    
    # We must ensure there is some document indexed. Let's build a quick index so it's not missing.
    docs = [
        {"text": "LoRA is a method for adapting pre-trained models by freezing original weights and introducing low-rank matrices to modify the model's behavior.", "source": "paper1"},
        {"text": "This approach reduces memory consumption and accelerates training compared to full model fine-tuning.", "source": "paper2"}
    ]
    rg.pipeline.retriever.build_from_documents(docs)
    init_latency = time.perf_counter() - start_init
    
    query = "What is LoRA?"
    
    start_run = time.perf_counter()
    result = rg.run(query)
    run_latency = time.perf_counter() - start_run
    
    # Assertions
    assert result.answer is not None and len(result.answer) > 0, "Answer does not exist"
    assert result.judgment is not None, "Judgment does not exist"
    assert result.judgment.total_claims > 0, "Claims were not extracted"
    assert result.evaluation is not None, "Evaluation does not exist"
    assert result.latency > 0, "Latency not recorded"
    assert result.judgment.faithfulness_score >= 0.8, f"Faithfulness score too low: {result.judgment.faithfulness_score}"
    assert result.repair_result is not None, "Repair result not populated"
    
    print(f"\nQUESTION:\n{result.query}\n")
    print(f"ANSWER:\n{result.answer}\n")
    print(f"CLAIMS EXTRACTED:\n{result.judgment.total_claims} claims\n")
    
    print(f"JUDGMENT:\nSupported: {result.judgment.supported} | Neutral: {result.judgment.neutral} | Contradicted: {result.judgment.contradicted}\n")
    print(f"REPAIR:\nNeeded: {result.judgment.repair_needed} | Attempts: {result.repair_result.attempt} | Strategy: {result.repair_result.strategy.value}\n")
    
    print(f"EVALUATION:\nFaithfulness: {result.evaluation.faithfulness:.2f} | Answer Relevancy: {result.evaluation.answer_relevancy:.2f}\n")
    
    print("LATENCY METRICS:")
    print(f"- Initialization (incl. indexing): {init_latency:.2f}s")
    print(f"- Pipeline Run Latency:            {result.latency:.2f}s")
    
    print("\n[SUCCESS] True End-to-End Validation Complete!")

if __name__ == "__main__":
    main()
