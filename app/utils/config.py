from pathlib import Path

# Dynamically find the absolute path to the 'app' directory
BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"

TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer"
ONNX_MODEL_PATH = ARTIFACTS_DIR / "model" / "model.onnx"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "encoder" / "label_encoder.joblib"
