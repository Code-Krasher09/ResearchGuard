import time
from typing import Optional
from loguru import logger
from src.schemas.query import Query
from src.schemas.repair import RepairPlan, RepairStrategy
from src.schemas.repair_result import RepairResult, RepairAttempt
from src.schemas.judgment import Judgment
from src.schemas.pipeline import PipelineComponents

MAX_REPAIR_ATTEMPTS = 3

class RepairExecutor:
    """
    Executes repair plans and manages the regeneration verification loop.
    """
    def execute(self, query: Query, repair_plan: RepairPlan, pipeline: PipelineComponents) -> RepairResult:
        attempt = 1
        current_plan = repair_plan
        start_time = time.perf_counter()
        
        judgment = None
        answer = None
        history = []
        
        while attempt <= MAX_REPAIR_ATTEMPTS:
            logger.info(f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS}: strategy={current_plan.strategy.value}")
            
            k = current_plan.new_k if current_plan.new_k is not None else current_plan.current_k
            
            # 1. Retrieve
            evidence = pipeline.retriever.retrieve(query.text, k=k)
            
            # 2. Generate
            answer_obj = pipeline.generator.generate_answer(query.text, evidence, rewrite_required=current_plan.rewrite_required)
            answer = answer_obj.answer
            
            # 3. Extract
            claims = pipeline.claim_extractor.extract_claims(answer)
            
            # 4. Verify
            verifications = pipeline.verifier.verify_batch(claims, evidence)
            
            # 5. Judge
            judgment = pipeline.judge.judge(claims, verifications)
            
            history.append(RepairAttempt(
                attempt=attempt,
                score=judgment.faithfulness_score,
                strategy=current_plan.strategy
            ))
            
            if not judgment.repair_needed:
                latency = time.perf_counter() - start_time
                return RepairResult(
                    attempt=attempt,
                    strategy=current_plan.strategy,
                    success=True,
                    judgment=judgment,
                    answer=answer,
                    latency=latency,
                    repair_history=history,
                    claims=claims,
                    verification_results=verifications
                )
                
            if pipeline.planner and attempt < MAX_REPAIR_ATTEMPTS:
                current_plan = pipeline.planner.plan(query, judgment, verifications, current_k=k)
                
            attempt += 1

        latency = time.perf_counter() - start_time
        return RepairResult(
            attempt=MAX_REPAIR_ATTEMPTS,
            strategy=current_plan.strategy,
            success=False,
            judgment=judgment,
            answer=answer,
            latency=latency,
            repair_history=history,
            claims=claims if 'claims' in locals() else None,
            verification_results=verifications if 'verifications' in locals() else None
        )
