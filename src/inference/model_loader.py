# src/inference/model_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, Any, Dict

import joblib

from src.common.config import settings
from src.common.logger import log
from src.common.preprocessing import TripPreprocessor


def load_active_model() -> Tuple[Any, TripPreprocessor, Dict[str, Any]]:
    """
    Загружает активную модель и KMeans из модельного регистра.

    Возвращает:
    - model: обученная модель (LightGBM/CatBoost)
    - preprocessor: TripPreprocessor с загруженными KMeans
    - metadata: dict с версией и метриками
    """
    registry_dir = Path(settings.MODEL_REGISTRY_PATH)
    registry_file = registry_dir / "registry.json"

    if not registry_file.exists():
        raise FileNotFoundError(f"registry.json not found at {registry_file}")

    with registry_file.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    active_version = registry.get("active_version")
    if not active_version:
        raise ValueError("No active_version in registry.json")

    versions = registry.get("versions", {})
    version_info = versions.get(active_version, {})

    version_dir = registry_dir / "versions" / active_version

    model_path = version_dir / "model.pkl"
    kmeans_pickup_path = version_dir / "kmeans_pickup.pkl"
    kmeans_dropoff_path = version_dir / "kmeans_dropoff.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not kmeans_pickup_path.exists() or not kmeans_dropoff_path.exists():
        raise FileNotFoundError("KMeans files not found in version directory")

    model = joblib.load(model_path)
    kmeans_pickup = joblib.load(kmeans_pickup_path)
    kmeans_dropoff = joblib.load(kmeans_dropoff_path)

    preprocessor = TripPreprocessor(n_clusters=settings.KMEANS_CLUSTERS)
    preprocessor.kmeans_pickup = kmeans_pickup
    preprocessor.kmeans_dropoff = kmeans_dropoff

    metadata = {
        "version": active_version,
        "metrics": version_info.get("metrics", {}),
        "updated_at": registry.get("updated_at"),
    }

    log.info(
        f"Loaded model and KMeans for version={active_version} "
        f"from {version_dir}"
    )

    return model, preprocessor, metadata
