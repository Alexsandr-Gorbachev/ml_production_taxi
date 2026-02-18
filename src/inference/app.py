# src/inference/app.py
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.common.config import settings
from src.common.logger import setup_logger, log
from src.common.schemas import TripRequest, TripResponse, HealthResponse
from src.inference.model_loader import load_active_model   # ✅ абсолютный
from src.inference.predictor import TripPredictor           # ✅ абсолютный


app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description="Inference service (Stage 4, full preprocessing + KMeans + log-target)",
    version="4.0.0",
)

# Глобальные объекты, переинициализируются при /model/reload
PREDICTOR: TripPredictor | None = None
MODEL_VERSION: str = "unknown"


def _init_model() -> None:
    """
    Загружает модель + KMeans и создаёт TripPredictor.
    Вызывается при старте и при POST /model/reload.
    """
    global PREDICTOR, MODEL_VERSION

    model, preprocessor, metadata = load_active_model()
    MODEL_VERSION = metadata.get("version", "unknown").strip()
    PREDICTOR = TripPredictor(
        model=model,
        preprocessor=preprocessor,
        version=MODEL_VERSION,
    )
    log.info(f"✅ Model initialized: version={MODEL_VERSION}")


@app.on_event("startup")
def startup_event() -> None:
    """Хук запуска: настраиваем логгер, загружаем модель."""
    setup_logger("inference")
    log.info("🚀 Starting inference service...")
    _init_model()
    log.info("✅ Inference service ready")


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """Базовый эндпоинт: статус и версия модели."""
    return HealthResponse(
        status="healthy" if PREDICTOR is not None else "unhealthy",
        model_loaded=PREDICTOR is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now(),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health-check для Docker/Kubernetes."""
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
    """Принимает TripRequest, возвращает предсказанную длительность."""
    if PREDICTOR is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return PREDICTOR.predict(trip)
    except Exception as e:
        log.exception(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/model/reload")
def reload_model() -> JSONResponse:
    """
    Перезагружает модель без перезапуска контейнера.
    Вызывается автоматически из Training сервиса после деплоя.
    """
    try:
        _init_model()
        log.info(f"🔄 Model reloaded: {MODEL_VERSION}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "message": "Model reloaded",
                "model_version": MODEL_VERSION,
            },
        )
    except Exception as e:
        log.exception(f"❌ Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload error: {e}")
    


