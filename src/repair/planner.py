from typing import List
from src.schemas.query import Query
from src.schemas.judgment import Judgment
from src.schemas.verification import VerificationResult
from src.schemas.repair import RepairPlan, RepairStrategy

class RepairPlanner:
    """
    Diagnoses verification failures and recommends an actionable repair strategy.
    """
    def plan(self, query: Query, judgment: Judgment, verification_results: List[VerificationResult], current_k: int = 5) -> RepairPlan:
        if not judgment.repair_needed:
            return RepairPlan(
                strategy=RepairStrategy.NONE,
                reason="No repair needed.",
                current_k=current_k,
                confidence=1.0
            )

        adaptive_k = max(5, current_k * 2)

        if judgment.total_claims == 0 or all(not res.evidence_chunk_ids for res in verification_results):
            return RepairPlan(
                strategy=RepairStrategy.HYBRID,
                reason="Empty evidence or claims",
                current_k=current_k,
                new_k=adaptive_k,
                rewrite_required=True,
                confidence=0.8
            )
            
        if judgment.contradicted > 0:
            return RepairPlan(
                strategy=RepairStrategy.QUERY_REWRITE,
                reason="Contradictions found. Query rewrite required.",
                current_k=current_k,
                rewrite_required=True,
                confidence=0.9
            )
            
        if judgment.reason == "Low support" or judgment.faithfulness_score < 1.0:
            return RepairPlan(
                strategy=RepairStrategy.INCREASE_K,
                reason="Low support score. Increasing context retrieval.",
                current_k=current_k,
                new_k=adaptive_k,
                confidence=0.85
            )
            
        return RepairPlan(
            strategy=RepairStrategy.HYBRID,
            reason="Unknown failure.",
            current_k=current_k,
            new_k=adaptive_k,
            rewrite_required=True,
            confidence=0.5
        )
