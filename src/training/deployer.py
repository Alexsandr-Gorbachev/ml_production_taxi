# src/training/deployer.py
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib

from src.common.config import settings
from src.common.logger import log
from src.common.preprocessing import TripPreprocessor


def deploy_model(
    model,
    preprocessor: TripPreprocessor,
    metrics: Dict[str, Any],
    version: str,
) -> None:
    """
    Сохраняет модель и KMeans в models/versions/{version},
    обновляет registry.json, копирует активные артефакты в корень models/.
    """
    registry_dir = Path(settings.MODEL_REGISTRY_PATH)
    versions_dir = registry_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    version_dir = versions_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # 1. Сохранение артефактов версии
    model_path = version_dir / "model.pkl"
    kmeans_pickup_path = version_dir / "kmeans_pickup.pkl"
    kmeans_dropoff_path = version_dir / "kmeans_dropoff.pkl"
    metrics_path = version_dir / "metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor.kmeans_pickup, kmeans_pickup_path)
    joblib.dump(preprocessor.kmeans_dropoff, kmeans_dropoff_path)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": version,
                "created_at": datetime.utcnow().isoformat(),
                **metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    log.info(f"Saved model and KMeans to {version_dir}")

    # 2. Обновление registry.json
    registry_file = registry_dir / "registry.json"
    if registry_file.exists():
        with registry_file.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {}

    versions = registry.get("versions", {})
    versions[version] = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }

    registry["active_version"] = version
    registry["updated_at"] = datetime.utcnow().isoformat()
    registry["versions"] = versions

    with registry_file.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    log.info(f"Updated registry.json, active_version={version}")

    # 3. Копируем активные артефакты в корень models/
    shutil.copy(model_path, registry_dir / "model.pkl")
    shutil.copy(kmeans_pickup_path, registry_dir / "kmeans_pickup.pkl")
    shutil.copy(kmeans_dropoff_path, registry_dir / "kmeans_dropoff.pkl")

    log.info("Copied active model and KMeans to models/ root")
