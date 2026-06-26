from pydantic import BaseModel, Field
from typing import List

class VerificationResult(BaseModel):
    claim_id: str
    claim: str
    evidence_chunk_ids: List[str]
    top_evidence: str
    label: str = Field(description="SUPPORTED, NEUTRAL, or CONTRADICTED")
    top_confidence: float = Field(ge=0.0, le=1.0)
