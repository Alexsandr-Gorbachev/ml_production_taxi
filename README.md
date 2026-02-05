# 🚕 NYC Taxi Trip Duration — ML Service

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-orange?logo=Yandex)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![RMSLE](https://img.shields.io/badge/RMSLE-0.351-brightgreen)

> Production-ready **MLOps pipeline** для предсказания длительности поездок NYC Taxi (CatBoost + Optuna + FastAPI + Docker Compose). Полный цикл: обучение → деплой → инференс → hot-reload.

---

## 📁 Структура проекта

```bash
ml-mvp/
├── .env                   # Конфиги (RMSLE_THRESHOLD=0.40)
├── docker-compose.yml     # training, inference:8000, tools
├── pyproject.toml         # fastapi, catboost, optuna, uv
├── data/newdata.csv       # 1.4M строк NYC Taxi
├── models/                # Registry (model.cb0, registry.json, versions/)
└── src/                   # common/, inference/, training/

🚀 Quick Start

bash
# 1. Установить uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Синхронизация
cd ml-mvp && uv sync

# 3. Обучить модель (619 итераций Optuna)
docker compose --profile tools up training

# 4. Запустить API
docker compose up inference

# 5. Тесты
uv run pytest tests  # ~80% coverage

Swagger UI: http://localhost:8000/docs
🧠 Обучение модели

Данные: data/newdata.csv (1.4M строк)
Команда: docker compose --profile tools up training

Pipeline:

    Haversine distance, datetime (hour, dayofweek, weekend)

    KMeans (pickup/dropoff, 10 clusters)

    Outliers p99.86, log(target)

    CatBoost (lr=0.145, depth=6, iterations=619)

Метрики: RMSLE=0.351, RMSE=348s, MAE=192s, R²=0.73
Валидация: RMSLE < 0.40 → deploy в models/versions/vNEW/
Логи: docker logs taxitraining
🔗 API Endpoints
Endpoint	Метод	Описание
/health	GET	Статус + model version
/predict	POST	Single prediction (~9ms)
/modelinfo	GET	RMSLE модели
/modelreload	POST	Hot‑reload из registry
Single Prediction

bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "pickuplongitude": -73.982,
  "pickuplatitude": 40.768,
  "dropofflongitude": -73.965,
  "dropofflatitude": 40.766,
  "passengercount": 1,
  "pickupdatetime": "2016-03-14 17:24:55"
}'

json
{"predicteddurationseconds":526.78,"predicteddurationminutes":8.78,"modelversion":"v20260205121526"}

Batch Predictions

bash
uv run python tools/batchpredict.py --input data/newdata.csv --output predictions.csv
# 1.4M строк → ~30s CPU

📊 Результаты
Метрика	Baseline	Optuna
RMSLE	0.368	0.351
RMSE	373s	348s
MAE	209s	192s
R²	0.690	0.731

Фичи: 30 (haversine, datetime, KMeans clusters)
🛤️ Roadmap

    ☑️ CatBoost + Optuna tuning

    ☑️ Model Registry + hot-reload

    ☑️ Docker Compose (training/inference)

    🔜 MLflow

    🔜 GitHub Actions CI/CD

    🔜 Prometheus monitoring

🛠️ Stack

ML: CatBoost, Optuna, scikit-learn
API: FastAPI
Infra: Docker Compose, uv
Config: Pydantic
Logging: Loguru

⭐ MLOps MVP — production-ready пример полного ML‑цикла на NYC Taxi dataset.
