from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dynamically find the absolute path to the 'app' directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost"]
    
    # ML Model Settings
    # We use pathlib to construct OS-agnostic paths relative to the app directory
    artifacts_dir: Path = BASE_DIR / "artifacts"
    
    @property
    def tokenizer_path(self) -> Path:
        return self.artifacts_dir / "tokenizer"
        
    @property
    def onnx_model_path(self) -> Path:
        return self.artifacts_dir / "model" / "model.onnx"
        
    @property
    def label_encoder_path(self) -> Path:
        return self.artifacts_dir / "encoder" / "label_encoder.joblib"
    
    # Pydantic configuration to load the .env file automatically
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate a single, global settings object
settings = Settings()
