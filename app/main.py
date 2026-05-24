import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from utils.lifespan import lifespan
from utils.logging import setup_logging
from api.routes import router
from api.schemas import HealthResponse
from utils.config import settings

setup_logging(log_level="INFO")

logger = logging.getLogger(__name__)

app = FastAPI(
    lifespan=lifespan,
    title="Ticket Triage API"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

@app.get("/")
@app.get("/health",summary="Health Check Endpoint", tags=["General"])
async def health_check(request: Request):
    """The Health Check Endpoint."""
    logger.info(msg="Health Check Requested.")
    model_status = request.app.state.classifier.model_status()
    return HealthResponse(
        status="ok",
        model_loaded = model_status["model"] is not None,
        label_encoder_loaded = model_status["label_encoder"] is not None,
        tokeniser_loaded = model_status["tokeniser"] is not None,
        model_version= "v1"
    )

if __name__ == "__main__":
    uvicorn.run(
        app=app,
        host=settings.api_host,
        port=settings.api_port
    )
