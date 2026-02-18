# src/training/deployer.py
import json
import shutil
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib

from src.common.config import settings
from src.common.logger import log
from src.common.preprocessing import TripPreprocessor


def notify_inference_reload(version: str) -> bool:
    """
    Уведомляет inference сервис о новой модели через POST /model/reload.
    Возвращает True если успешно, False если сервис недоступен.
    """
    try:
        url = f"{settings.INFERENCE_HOST}/model/reload"  # ✅ полный URL из settings
        log.info(f"🔔 Notifying inference service: {url}")
        response = requests.post(url, timeout=30)         # ✅ убрал params — не нужен

        if response.status_code == 200:
            log.info(f"✅ Inference reloaded → {version}")
            return True
        else:
            log.error(f"❌ Inference reload failed: HTTP {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        log.warning("⚠️ Inference service unavailable — reload it manually: POST /model/reload")
        return False
    except Exception as e:
        log.error(f"❌ Could not notify inference service: {e}")
        return False


def deploy_model(
    model,
    preprocessor: TripPreprocessor,
    metrics: Dict[str, Any],
    version: str,
) -> bool:
    """
    Сохраняет модель и KMeans в models/versions/{version},
    обновляет registry.json, копирует активные артефакты в корень models/,
    уведомляет inference сервис о перезагрузке.
    Возвращает True если деплой и reload прошли успешно.
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

    log.info(f"✅ Saved model and KMeans to {version_dir}")

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

    log.info(f"✅ Updated registry.json, active_version={version}")

    # 3. Копируем активные артефакты в корень models/
    shutil.copy(model_path, registry_dir / "model.pkl")
    shutil.copy(kmeans_pickup_path, registry_dir / "kmeans_pickup.pkl")
    shutil.copy(kmeans_dropoff_path, registry_dir / "kmeans_dropoff.pkl")

    log.info("✅ Copied active model and KMeans to models/ root")

    # 4. ✅ Уведомляем inference сервис о новой модели
    reloaded = notify_inference_reload(version)
    if not reloaded:
        log.warning(
            "⚠️ Inference не перезагружен автоматически. "
            "Вызови вручную: POST /model/reload"
        )

    return reloaded
