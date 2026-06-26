import time
from loguru import logger
from src.schemas.query import Query
from src.generation.client import GroqClient
from src.generation.generator import Generator
from src.generation.prompts import PromptManager
from src.retrieval.embedder import Embedder
from src.retrieval.chunker import Chunker
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.retriever import Retriever
from src.verification.claims import ClaimExtractor
from src.verification.verifier import Verifier
from src.verification.judge import Judge
from src.evaluation.evaluator import Evaluator

def main():
    logger.info("Initializing End-to-End Smoke Test")
    
    docs = [
        {"text": "LoRA is a method for adapting pre-trained models by freezing original weights and introducing low-rank matrices to modify the model's behavior.", "source": "paper1"},
        {"text": "This approach reduces memory consumption and accelerates training compared to full model fine-tuning.", "source": "paper2"}
    ]
    
    query_text = "What is LoRA?"
    
    # Init Pipeline
    retriever = Retriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embedder = retriever.embedder
    
    retriever.build_from_documents(docs)
    
    client = GroqClient()
    pm = PromptManager()
    generator = Generator(client=client, prompt_manager=pm)
    claim_extractor = ClaimExtractor()
    verifier = Verifier()
    judge = Judge()
    evaluator = Evaluator(embedder=embedder)
    
    print("\n==============================================")
    print("      RESEARCHGUARD SMOKE TEST EXECUTION       ")
    print("==============================================\n")
    
    print(f"QUESTION:\n{query_text}\n")
    
    # 1. Retrieve
    evidence = retriever.retrieve(query_text, k=2)
    
    # 2. Generate
    answer_obj = generator.generate_answer(query_text, evidence)
    answer = answer_obj.answer
    print(f"ANSWER:\n{answer}\n")
    
    # 3. Claims
    claims = claim_extractor.extract_claims(answer)
    print("CLAIMS:")
    for c in claims:
        print(f" - [{c.id}] {c.text}")
    print()
    
    # 4. Verification
    verifications = verifier.verify_batch(claims, evidence)
    print("VERIFICATION:")
    for v in verifications:
        print(f" - [{v.claim_id}] {v.label} (Conf: {v.top_confidence:.4f}) -> {v.top_evidence}")
    print()
    
    # 5. Judgment
    judgment = judge.judge(claims, verifications)
    print(f"JUDGMENT:\nSupported: {judgment.supported} | Neutral: {judgment.neutral} | Contradicted: {judgment.contradicted}\n")
    
    # Faithfulness (from Judgment)
    print(f"FAITHFULNESS SCORE: {judgment.faithfulness_score:.2f}\n")
    
    # Evaluator Faithfulness (Fallback heuristic)
    eval_result = evaluator.evaluate(question=query_text, answer=answer, contexts=[e.text for e in evidence], ground_truth=None, repair_triggered=judgment.repair_needed)
    print(f"EVALUATOR METRICS: Faithfulness={eval_result.faithfulness:.2f}, Answer Relevancy={eval_result.answer_relevancy:.2f}\n")
    
    # 6. Repair
    print(f"REPAIR NEEDED: {judgment.repair_needed}")
    if judgment.repair_needed:
        print(f"REASON: {judgment.reason}")

if __name__ == "__main__":
    main()
