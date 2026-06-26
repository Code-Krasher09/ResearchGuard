from pydantic import BaseModel

class Claim(BaseModel):
    id: str
    text: str
    position: int
    claim_type: str = "FACTUAL"
