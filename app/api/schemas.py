from typing import Dict, Optional, List

from pydantic import BaseModel, Field, field_validator

class TicketRequest(BaseModel):
    """Single ticket classification request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Ticket description text",
    )
    return_all_scores: Optional[bool]
   
    @field_validator('text')
    @classmethod
    def validate_ticket(cls, value: str) -> str:
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

class BatchTicketRequest(BaseModel):
    """Batch ticket classification request."""
    tickets: List[str] = Field(
        min_length=1,
        max_length=2048,
        description="List of ticket texts",
        examples=[["Laptop broken", "Need access to drive"]]
    )
    
    @field_validator('tickets')
    @classmethod
    def validate_tickets(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Ticket list cannot be empty")
        # Strip whitespace
        return [text.strip() for text in v if text.strip()]
        
class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[str] = Field(description="Predicted ticket category")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="API health status")
    model_loaded: bool = Field(description="Whether Model is loaded in memory")
    label_encoder_loaded: bool = Field(description="Whether Label Encoder is loaded in memory")
    tokeniser_loaded: bool = Field(description="Whether Tokeniser is loaded in memory")
    model_version: str = Field(description="Model version")
