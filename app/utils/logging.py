import sys
import logging

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level.upper(),
        format=log_format,
        handlers=[ 
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set uvicorn & transformers logging to Warning to reduce clutter
    logging.getLogger('uvicorn').setLevel(logging.WARNING)   
    logging.getLogger('transformers').setLevel(logging.WARNING)
   