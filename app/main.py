import logging
from ui import demo
from utils.logging import setup_logging

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", server_port=7860)
