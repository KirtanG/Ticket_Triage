import logging

import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TicketClassifier:

    def __init__(self) -> None:
        self.tokeniser = None
        self.ort_session = None
        self._load_model() 

    def _load_model(self) -> None:
        try:
            self.tokeniser = AutoTokenizer.from_pretrained(r"./app/artifacts",local_files_only=True)
            self.ort_session = ort.InferenceSession(path_or_bytes=r"./app/artifacts/model.onnx")
        except Exception as e:
            logger.error(f"Failed to load model!")
        
    def _tokenise(self):
        """
        Tokenises the input.
        """
