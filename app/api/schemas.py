from typing import Dict,List, Optional

from pydantic import BaseModel, Field, field_validator

class TicketRequest(BaseModel):
    """Single ticket classification request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Ticket description text",
    )

   
    @field_validator('text')
    @classmethod
    def validate(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return value.strip()
