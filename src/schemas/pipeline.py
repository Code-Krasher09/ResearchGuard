from pydantic import BaseModel, ConfigDict
from typing import Any

class PipelineComponents(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    retriever: Any
    generator: Any
    claim_extractor: Any
    verifier: Any
    judge: Any
    planner: Any
