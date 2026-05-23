import logging

import uvicorn
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware

from utils.lifespan import lifespan
from utils.logging import setup_logging
from api.routes import router
from api.schemas import HealthResponse

setup_logging(log_level="INFO")

logger = logging.getLogger(__name__)

app = FastAPI(
    lifespan=lifespan,
    title="Ticket Triage API"
)

allowed_origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=router)

@app.get("/")
@app.get("/health",summary="Health Check Endpoint",tags=["General"])
async def health_check(request: Request):
    """ The Health Check Endpoint."""
    logger.info(msg="Health Check Requested.")
    model_status = request.app.state.classifier.model_status()
    return HealthResponse(
        status="ok",
        model_loaded = True if model_status["model"] is not None else False,
        label_encoder_loaded = True if model_status["label_encoder"] is not None else False,
        tokeniser_loaded = True if model_status["tokeniser"] is not None else False,
        model_version= "v1"
    )

if __name__ == "__main__":
    uvicorn.run(app=app,host="0.0.0.0",port=8080)
