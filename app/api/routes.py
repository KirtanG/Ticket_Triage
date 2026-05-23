import logging

from fastapi import APIRouter, Request, status
from fastapi.exceptions import HTTPException

from api.schemas import TicketRequest , PredictionResponse , BatchPredictionResponse , BatchTicketRequest    

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["Version 1 Prediction Endpoints"]
)

@router.post("/predict",summary="Classify Single Ticker")
async def predict_online( ticket: TicketRequest , request: Request):
    model = request.app.state.classifier
    
    if not model.is_loaded():
        logger.error("The model isn't loaded during prediction time!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        text = ticket.text.lower()

        predicted_class, confidence, all_scores  = model.predict_single(text)

        return PredictionResponse(
            predicted_class = predicted_class,
            confidence = confidence,
            all_scores = all_scores
        )

    except Exception as e:

        logger.error(msg="An Exception occured in Online Prediction API"+str(e))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed!"
        )

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify multiple tickets",
)
async def predict_batch(batch_tickets:BatchTicketRequest , request: Request):
    """
    Classify multiple IT support tickets in a single request.
    
    More efficient than multiple single predictions.
    Maximum 100 tickets per request.
    """
    model = request.app.state.classifier
    
    if not model.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        predictions  = model.predict_batch(batch_tickets.tickets)
        
        batch_predictions = [
            pred_class
            for text, (pred_class, conf) in zip(batch_tickets.tickets, predictions)
        ]
        
        return BatchPredictionResponse(
            predictions=batch_predictions
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )    





