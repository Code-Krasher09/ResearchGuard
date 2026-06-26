from pydantic import BaseModel, Field
from typing import Optional

class GeneratedAnswer(BaseModel):
    answer: str
    model: str
    latency: float = Field(ge=0.0)
    estimated_tokens: int = Field(ge=0)
    prompt_version: Optional[str] = None
