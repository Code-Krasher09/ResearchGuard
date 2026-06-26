from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import datetime
import uuid

class Query(BaseModel):
    """
    Represents a user input query in ResearchGuard.
    """
    id: str = Field(..., description="Unique identifier, preferably a UUID")
    text: str = Field(..., min_length=1, description="The user question text")
    timestamp: datetime.datetime = Field(..., description="ISO format timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @classmethod
    def create(cls, text: str, query_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> "Query":
        """
        Helper to create a query with an auto-generated ISO timestamp and optional UUID.
        """
        return cls(
            id=query_id if query_id else str(uuid.uuid4()),
            text=text,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            metadata=metadata or {}
        )
