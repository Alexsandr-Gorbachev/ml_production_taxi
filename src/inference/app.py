# src/inference/app.py
from __future__ import annotations

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.common.config import settings
from src.common.logger import setup_logger, log
from src.common.schemas import TripRequest, TripResponse, HealthResponse
from .model_loader import load_active_model
from .predictor import TripPredictor

app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Inference service (Stage 4, full preprocessing + KMeans + log-target)",
    version="4.0.0",
)

# Глобальные объекты, которые будут переинициализироваться при reload
PREDICTOR: TripPredictor | None = None
MODEL_VERSION: str = "unknown"


def _init_model() -> None:
    """
    Внутренняя функция: загрузить модель + KMeans и создать TripPredictor.
    Вызывается при старте и при /model/reload.
    """
    global PREDICTOR, MODEL_VERSION

    model, preprocessor, metadata = load_active_model()
    MODEL_VERSION = metadata.get("version", "unknown")
    PREDICTOR = TripPredictor(model=model, preprocessor=preprocessor, version=MODEL_VERSION)

    log.info(f"Inference initialized with model version={MODEL_VERSION}")


@app.on_event("startup")
def startup_event() -> None:
    """
    Хук запуска приложения:
    - настраиваем логгер,
    - загружаем модель и KMeans,
    - создаём TripPredictor.
    """
    setup_logger("inference")
    log.info("Starting inference service...")
    _init_model()
    log.info("Inference service is ready")


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """
    Базовый эндпоинт: возвращает статус и текущую версию модели.
    """
    return HealthResponse(
        status="healthy" if PREDICTOR is not None else "unhealthy",
        model_loaded=PREDICTOR is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health-check для Docker/Kubernetes.
    """
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(),
    )


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest) -> TripResponse:
    """
    Основной эндпоинт: принимает TripRequest и возвращает предсказанную длительность.
    """
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return PREDICTOR.predict(trip)
    except Exception as e:
        log.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/model/reload")
def reload_model():
    """
    Эндпоинт для Training сервиса:
    - перечитывает registry.json,
    - загружает новую модель + KMeans,
    - обновляет глобальный PREDICTOR.
    """
    try:
        _init_model()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Model reloaded",
                "model_version": MODEL_VERSION,
            },
        )
    except Exception as e:
        log.exception(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload error: {e}")


