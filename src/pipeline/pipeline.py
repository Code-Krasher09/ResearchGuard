import time
from typing import Optional
from loguru import logger
from src.schemas.query import Query
from src.schemas.repair import RepairPlan, RepairStrategy
from src.schemas.pipeline import PipelineComponents
from src.schemas.pipeline_result import PipelineResult
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
from src.repair.planner import RepairPlanner
from src.repair.executor import RepairExecutor
from src.evaluation.evaluator import Evaluator

class ResearchGuard:
    """
    The main entrypoint for the ResearchGuard Pipeline.
    Initializes all core components and orchestrates the end-to-end RAG workflow.
    """
    def __init__(self, pipeline: Optional[PipelineComponents] = None, evaluator: Optional[Evaluator] = None):
        if pipeline:
            self.pipeline = pipeline
        else:
            logger.info("Initializing ResearchGuard default pipeline components...")
            retriever = Retriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            prompt_manager = PromptManager()
            client = GroqClient()
            generator = Generator(client=client, prompt_manager=prompt_manager)
            
            claim_extractor = ClaimExtractor()
            verifier = Verifier()
            judge = Judge()
            planner = RepairPlanner()
            
            self.pipeline = PipelineComponents(
                retriever=retriever,
                generator=generator,
                claim_extractor=claim_extractor,
                verifier=verifier,
                judge=judge,
                planner=planner
            )
            
        self.executor = RepairExecutor()
        self.evaluator = evaluator or Evaluator(embedder=getattr(self.pipeline.retriever, 'embedder', None))

    def run(self, query_text: str, ground_truth: Optional[str] = None) -> PipelineResult:
        logger.info(f"Running ResearchGuard pipeline for query: '{query_text}'")
        start_time = time.perf_counter()
        
        query = Query.create(text=query_text)
        initial_k = 5
        
        initial_plan = RepairPlan(
            strategy=RepairStrategy.NONE,
            reason="Initial execution",
            current_k=initial_k,
            confidence=1.0
        )
        
        # 1. Execute Self-Healing Generation Loop
        repair_result = self.executor.execute(query, initial_plan, self.pipeline)
        
        # 2. Evaluate Final Quality
        contexts = [c.text for c in self.pipeline.retriever.retrieve(query.text, k=initial_k)]
        repairs_triggered = repair_result.attempt > 1
        
        evaluation = self.evaluator.evaluate(
            question=query.text,
            answer=repair_result.answer or "No answer generated.",
            contexts=contexts,
            ground_truth=ground_truth,
            repair_triggered=repairs_triggered
        )
        
        latency = time.perf_counter() - start_time
        logger.info(f"Pipeline finished in {latency:.2f}s. Success={repair_result.success}")
        
        return PipelineResult(
            query=query.text,
            answer=repair_result.answer or "No answer generated.",
            judgment=repair_result.judgment,
            repair_result=repair_result,
            evaluation=evaluation,
            latency=latency
        )
