"""
Configuration settings for ML Service (NYC Taxi MVP Этап 4)
Pydantic v2 + pydantic_settings + .env
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """All configuration from .env"""
    
    # === Model / Registry ===
    MODEL_PATH: str = "models/model.pkl"
    MODEL_REGISTRY_PATH: str = "models"
    
    # === Training Data / Split ===
    NEW_DATA_FILE: str = "data/new_data.csv"
    TRAIN_TEST_SPLIT: float = 0.8
    RANDOM_STATE: int = 42
    
    # === Этап 4: KMeans + Outliers ===
    KMEANS_CLUSTERS: int = 10
    TRIP_DURATION_UPPER_PERCENTILE: float = 99.86
    
    # === Metrics thresholds ===
    MIN_RMSLE_THRESHOLD: float = 0.40
    
    # === Server ===
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # === Logging ===
    LOG_LEVEL: str = "INFO"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # 🔥 Фиксит extra_forbidden!
    )

# Singleton instance
settings = Settings()

if __name__ == "__main__":
    """Debug: проверка загрузки"""
    print("✅ Settings loaded:")
    print(f"  Model registry: {settings.MODEL_REGISTRY_PATH}")
    print(f"  Data file: {settings.NEW_DATA_FILE}")
    print(f"  KMeans clusters: {settings.KMEANS_CLUSTERS}")
    print(f"  RMSLE threshold: {settings.MIN_RMSLE_THRESHOLD}")
    print(f"  Host/Port: {settings.HOST}:{settings.PORT}")

