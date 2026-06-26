from pydantic import BaseModel, Field

class Judgment(BaseModel):
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    supported: int = Field(ge=0)
    neutral: int = Field(ge=0)
    contradicted: int = Field(ge=0)
    total_claims: int = Field(ge=0)
    repair_needed: bool
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
