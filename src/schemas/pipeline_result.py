from pydantic import BaseModel, ConfigDict
from src.schemas.judgment import Judgment
from src.schemas.repair_result import RepairResult
from src.schemas.evaluation import EvaluationResult

class PipelineResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    query: str
    answer: str
    judgment: Judgment
    repair_result: RepairResult
    evaluation: EvaluationResult
    latency: float
