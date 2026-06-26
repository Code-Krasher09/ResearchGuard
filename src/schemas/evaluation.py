from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    repair_rate: float = Field(ge=0.0, le=1.0)
