from pydantic import BaseModel
from typing import Optional
from enum import Enum

class RepairStrategy(str, Enum):
    NONE = "NONE"
    INCREASE_K = "INCREASE_K"
    QUERY_REWRITE = "QUERY_REWRITE"
    HYBRID = "HYBRID"

class RepairPlan(BaseModel):
    strategy: RepairStrategy
    reason: str
    current_k: int
    new_k: Optional[int] = None
    rewrite_required: bool = False
    confidence: float
