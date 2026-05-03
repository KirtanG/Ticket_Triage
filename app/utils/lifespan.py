import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.ml.model import TicketClassifier

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    This Function handles the Startup and Shutdown Events.
    """
    logger.info("Starting the API server...")

    app.state.classifier = TicketClassifier()
    logger.info("Loaded the models successfully...")

    yield 
    
    logger.info("Shutting the server down...")

    app.state.classifier = None
