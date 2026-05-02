from typing import Dict,Optional

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
    
class PredictionResponse(BaseModel):
    """Single prediction response."""
    predicted_class: str = Field(description="Predicted ticket category")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)
    all_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probability scores for all classes"
    )
    model_version: str = Field(description="Model version identifier")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="API health status")
    model_loaded: bool = Field(description="Whether model is loaded in memory")
    model_version: str = Field(description="Model version")
