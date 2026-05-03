import logging
from typing import List, Dict , Tuple , Union

import joblib
import numpy as np
from onnxruntime import InferenceSession
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, ModernBertConfig

logger = logging.getLogger(__name__)

class TicketClassifier:

    def __init__(self) -> None:
        self.tokeniser : Union[None, ModernBertConfig] = None
        self.ort_session : Union[None, InferenceSession] = None
        self.label_encoder : Union[None , LabelEncoder] = None 
        self._load_model() 

    def _load_model(self) -> None:
        """
        Handles model loading and inference.
        """
        try:
            self.tokeniser = AutoTokenizer.from_pretrained(r"./app/artifacts",local_files_only=True)
            self.ort_session = InferenceSession(path_or_bytes=r"./app/artifacts/model.onnx")
            self.label_encoder = joblib.load(filename=r"./app/artifacts/encoder/label_encoder.joblib")
            print(type(self.tokeniser))
        except Exception as e:
            logger.error(f"Failed to load model!")
            raise
        
    def _tokenize(self,texts: List[str] ) -> Dict[str, np.ndarray]:
        """
        Tokenises the input.
        """
        inputs = self.tokeniser(
            texts,
            padding = True,
            truncation = True,
            max_length = 256,
            return_tensors='np'
        )

        return {
            "input_ids" : inputs['input_ids'].astype(np.int64),
            "attention_mask": inputs['attention_mask'].astype(np.int64) 
        }
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def predict_single(
        self, 
        text: str, 
        return_all_scores: bool = True
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict class for a single ticket.
        
        Returns:
            (predicted_class, confidence, all_scores, inference_time_ms)
        """
        
        # Tokenize
        ort_inputs = self._tokenize([text])
        
        # Inference
        logits = self.ort_session.run(None, ort_inputs)[0]
        probs = self._softmax(logits)[0]
        
        # Get prediction
        pred_class_id = int(np.argmax(probs))
        
        # Decode using label encoder
        predicted_class = self.label_encoder.inverse_transform([pred_class_id])[0]
        confidence = float(probs[pred_class_id])
        
        # All scores (with decoded class names)
        all_scores = None
        if return_all_scores:
            all_scores = {
                self.label_encoder.inverse_transform([i])[0]: float(probs[i]) 
                for i in range(len(self.label_encoder.classes_))
            }
        
        
        logger.info(
            f"Prediction: {predicted_class} "
            f"(confidence: {confidence:.3f}"
        )
        
        return predicted_class, confidence, all_scores
    
    def predict_batch(
        self, 
        texts: List[str]
    ) -> Tuple[List[Tuple[str, float]]]:
        """
        Predict classes for multiple tickets.
        
        Returns:
            (list of (predicted_class, confidence), total_inference_time_ms)
        """
        
        # Tokenize batch
        ort_inputs = self._tokenize(texts)
        
        # Batch inference
        logits = self.ort_session.run(None, ort_inputs)[0]
        probs = self._softmax(logits)
        
        # Extract predictions and decode
        predictions = []
        for prob in probs:
            pred_class_id = int(np.argmax(prob))
            predicted_class = self.label_encoder.inverse_transform([pred_class_id])[0]
            confidence = float(prob[pred_class_id])
            predictions.append((predicted_class, confidence))
        
        logger.info(
            f"Batch prediction: {len(texts)} tickets "
        )
        
        return predictions
    
    def is_loaded(self) -> bool:
        """Check if model and label encoder are loaded."""
        return (
            self.ort_session is not None 
            and self.tokeniser is not None 
            and self.label_encoder is not None
        )
    
    def model_status(self) -> Dict[str,bool]:
        return {
            "model": self.ort_session is not None,
            "tokeniser": self.tokeniser is not None,
            "label_encoder": self.label_encoder is not None
        }

    def get_classes(self) -> List[str]:
        """Get list of all class labels."""
        return self.label_encoder.classes_.tolist()