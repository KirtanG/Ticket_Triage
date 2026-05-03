import logging

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health",summary="Health Check Endpoint",tags=["General"])
async def health_check(request: Request):
    """ The Health Check Endpoint."""
    model_status = request.app.state.classifier.model_status()
    return {
        "status" : "ok",
        **model_status
    }


