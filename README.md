🚕 NYC Taxi Trip Duration — MLOps Pipeline

    Production‑ready MLOps сервис для предсказания длительности поездок NYC Taxi на основе CatBoost + Optuna + FastAPI + Docker Compose.
    Полный цикл: обучение → деплой → инференс → hot‑reload моделей.

📁 Структура проекта

bash
ml-mvp/
├── .env                   # Конфиги (пути, порты, RMSLE_THRESHOLD=0.40)
├── .gitignore             # venv/, *.pyc, артефакты моделей и т.п.
├── docker-compose.yml     # 3 сервиса: training, inference (port 8000), tools
├── pyproject.toml         # Зависимости (FastAPI, CatBoost, Optuna, Pandas, uv)
├── uv.lock                # Lockfile для reproducible installs
├── data/
│   └── newdata.csv        # 1.4M строк NYC Taxi
├── models/
│   ├── model.cb0          # Активная CatBoost (symlink)
│   ├── kmeans.pkl         # Активный KMeans (symlink)
│   ├── registry.json      # Список версий и метрики (RMSLE=0.351)
│   └── versions/
│       └── v20260205121526/
│           ├── model.cb0
│           ├── kmeans.pkl
│           └── metrics.json
└── src/
    ├── common/            # Shared логика
    │   ├── config.py      # Pydantic settings (.env)
    │   ├── logger.py      # Loguru (INFO/ERROR)
    │   ├── preprocessing.py # TripPreprocessor (haversine, KMeans, outliers, log‑target)
    │   └── schemas.py     # Pydantic (TripRequest, PredictionResponse)
    ├── inference/
    │   ├── app.py         # FastAPI endpoints: /health, /predict, /modelinfo, /modelreload
    │   ├── modelloader.py # Загрузка моделей из registry.json
    │   └── predictor.py   # Препроцессинг → predict → expm1
    └── training/
        ├── train.py       # Обучение c Optuna tuning
        ├── validator.py   # Проверка RMSLE < 0.40
        └── deployer.py    # Сохранение модели, обновление registry.json

⚙️ Установка и запуск

1. Установить uv:

bash
curl -LsSf https://astral.sh/uv/install.sh | sh

2. Синхронизация зависимостей:

bash
cd ml-mvp
uv sync

3. Обучение модели:

bash
docker compose --profile tools up training

После 619 итераций Optuna сохранит новую версию модели в models/versions/vNEW/.

4. Запустить инференс:

bash
docker compose up inference

API будет доступно на → http://localhost:8000/docs

5. Тестирование:

bash
uv run pytest tests

(Покрытие ~80%: preprocessing, inference, models)
🧠 Обучение модели

Данные: data/newdata.csv (1.4M строк: pickup/dropoff координаты, пассажиры, datetime).
Запуск:

bash
docker compose --profile tools up training

Pipeline:

    препроцессинг: haversine distance, datetime (hour, dayofweek, weekend), KMeans=10

    фильтрация выбросов (p99.86 длительности), лог‑таргет

    Optuna tuning CatBoost (lr=0.145, depth=6, iterations=619)

    метрики:

        RMSLE = 0.351

        RMSE = 348s

        MAE = 192s

        R² = 0.73

Если RMSLE ≥ 0.40 → модель отклоняется.
При успехе деплой сохраняет артефакты и обновляет registry.json.

📜 Логи:

bash
docker logs taxitraining

🚀 Инференс API

Health Check

bash
curl http://localhost:8000/health
# {"status":"healthy","modelversion":"v20260205121526"}

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

Ответ (~9 ms):

json
{
  "predicteddurationseconds": 526.78,
  "predicteddurationminutes": 8.78,
  "modelversion": "v20260205121526"
}

Model info

bash
curl http://localhost:8000/modelinfo
# RMSLE=0.351

Hot reload

bash
curl -X POST http://localhost:8000/modelreload

Swagger UI: http://localhost:8000/docs
🧰 Batch‑предсказания

Через CLI:

bash
uv run python tools/batchpredict.py \
  --input data/newdata.csv \
  --output predictions.csv

Через Docker:

bash
docker compose run tools python tools/batchpredict.py \
  --input data/newdata.csv

Результат: predictions.csv (1.4M строк, latency ~30 s CPU).
Поддерживаются данные без tripduration и id.

PowerShell‑версия: batchpredict.ps1
📊 Метрики
Метрика	Baseline	Optuna Tuning
RMSLE	0.368	0.351
RMSE	373 s	348 s
MAE	209 s	192 s
R²	0.690	0.731

Фичи (30): haversine distance, datetime признаки, KMeans‑кластеры.
🔭 Roadmap

    ☑️ CatBoost модель с Optuna

    ☑️ Model Registry + hot reload

    🔜 MLflow Tracking

    🔜 CI/CD (GitHub Actions)

    🔜 Prometheus + Grafana мониторинг

🧠 Технологический стек
Категория	Используется
ML	CatBoost, Optuna, Pandas, scikit‑learn
API	FastAPI
Infra	Docker Compose, uv
Config	Pydantic settings
Logging	Loguru
Versioning	JSON‑based Model Registry
👤 Автор

MLOps MVP Project
→ Демонстрация продакшн‑ориентированного ML‑сервиса:
модульность, воспроизводимость, удобство CI/CD и версионности моделей.

Планы развития:

    Интеграция с MLflow для трекинга экспериментов

    CI/CD (GitHub Actions)

    Prometheus + Grafana для мониторинга моделей

🧰 Технологический стек

    ML: CatBoost, Optuna, pandas, scikit-learn

    API: FastAPI

    Infra: Docker Compose, uv

    Logging: Loguru

    Config: Pydantic settings

    Registry: JSON-based versioning system

🤖 Автор проекта

MLOps MVP — экспериментальный проект для демонстрации полного цикла MLOps на продакшн-примере NYC Taxi Duration.
Создан с упором на reproducibility, modularity и extensibility.
