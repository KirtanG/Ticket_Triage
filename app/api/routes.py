import logging

from fastapi import APIRouter, Request, status
from fastapi.exceptions import HTTPException

from api.schemas import TicketRequest , PredictionResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health",summary="Health Check Endpoint",tags=["General"])
async def health_check(request: Request):
    """ The Health Check Endpoint."""
    logger.info(msg="Health Check Requested.")
    model_status = request.app.state.classifier.model_status()
    return {
        "status" : "ok",
        **model_status
    }

@router.post("/predict",summary="Classify Single Ticker",tags=["Prediction"])
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

        HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed!"
        )
    





