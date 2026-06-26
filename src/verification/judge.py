from typing import List
from src.schemas.claim import Claim
from src.schemas.verification import VerificationResult
from src.schemas.judgment import Judgment

class Judge:
    """
    Evaluates faithfulness of claims based on verification results.
    Triggers repair if faithfulness score is below threshold or if there are contradictions.
    """
    def __init__(self, threshold: float = 0.8):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        
    def judge(self, claims: List[Claim], verification_results: List[VerificationResult]) -> Judgment:
        if not claims or not verification_results:
            return Judgment(
                faithfulness_score=0.0,
                supported=0,
                neutral=0,
                contradicted=0,
                total_claims=0,
                repair_needed=True,
                reason="Empty claims or verification results",
                confidence=0.0
            )

        claim_types = {c.id: c.claim_type for c in claims}
        factual_results = [
            r for r in verification_results 
            if claim_types.get(r.claim_id, "FACTUAL") != "REFUSAL"
        ]

        if not factual_results:
            # All claims were REFUSAL
            return Judgment(
                faithfulness_score=1.0,
                supported=0,
                neutral=0,
                contradicted=0,
                total_claims=len(claims),
                repair_needed=False,
                reason="Safe refusal",
                confidence=1.0
            )

        supported = 0
        neutral = 0
        contradicted = 0
        total_confidence = 0.0
        
        for res in factual_results:
            if res.label == "SUPPORTED":
                supported += 1
            elif res.label == "CONTRADICTED":
                contradicted += 1
            else:
                neutral += 1
                
            total_confidence += res.top_confidence
            
        total = len(factual_results)
        faithfulness_score = (supported + 0.5 * neutral) / total if total > 0 else 0.0
        avg_confidence = total_confidence / total if total > 0 else 0.0
        
        repair_needed = False
        reason = "Faithful"
        
        if contradicted > 0:
            repair_needed = True
            reason = "Contradiction found"
        elif faithfulness_score < self.threshold:
            repair_needed = True
            reason = "Low support"
            
        return Judgment(
            faithfulness_score=faithfulness_score,
            supported=supported,
            neutral=neutral,
            contradicted=contradicted,
            total_claims=len(claims),
            repair_needed=repair_needed,
            reason=reason,
            confidence=avg_confidence
        )
