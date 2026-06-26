from typing import List, Optional
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    """
    Represents retrieved scientific evidence chunk.
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., min_length=1, description="Chunk content text")
    source: str = Field(..., description="Source paper name or identifier")
    score: float = Field(..., ge=-1.0, le=1.0, description="Similarity score")
    embedding: Optional[List[float]] = Field(default=None, description="Optional dense embedding vector")
    section: Optional[str] = Field(default=None, description="Section of the document the chunk belongs to")
    page: Optional[int] = Field(default=None, description="Page number of the document")
