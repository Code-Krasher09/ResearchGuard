from pydantic import BaseModel
from typing import Optional, List, Any
from src.schemas.repair import RepairStrategy
from src.schemas.judgment import Judgment

class RepairAttempt(BaseModel):
    attempt: int
    score: float
    strategy: RepairStrategy

class RepairResult(BaseModel):
    attempt: int
    strategy: RepairStrategy
    success: bool
    judgment: Optional[Judgment] = None
    answer: Optional[str] = None
    latency: float
    repair_history: Optional[List[RepairAttempt]] = None
    claims: Optional[List[Any]] = None
    verification_results: Optional[List[Any]] = None
