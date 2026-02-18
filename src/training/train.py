# src/training/train.py
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor          # ✅ импорт наверху
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.common.config import settings
from src.common.logger import log, setup_logger
from src.common.preprocessing import TripPreprocessor
from src.training.deployer import deploy_model
from src.training.validator import validate_model


def train_model() -> None:
    """Full training pipeline — Stage 4"""
    setup_logger("training")
    log.info("=== STARTING TRAINING PIPELINE (Stage 4) ===")

    # 1. Load data
    data_path = Path(settings.NEW_DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    df = pd.read_csv(data_path)
    log.info(f"Loaded {len(df):,} rows from {data_path}")

    # 2. Outlier filtering
    df = df[df["trip_duration"] >= 60]

    upper_bound = df["trip_duration"].quantile(
        settings.TRIP_DURATION_UPPER_PERCENTILE / 100.0
    )
    df = df[df["trip_duration"] <= upper_bound]
    log.info(
        f"After duration filtering: {len(df):,} rows "
        f"(upper p{settings.TRIP_DURATION_UPPER_PERCENTILE} -> {upper_bound:.2f}s)"
    )

    df = df[
        (df["pickup_latitude"].between(40.5, 40.9)) &
        (df["dropoff_latitude"].between(40.5, 40.9)) &
        (df["pickup_longitude"].between(-74.3, -73.7)) &
        (df["dropoff_longitude"].between(-74.3, -73.7))
    ]
    df = df[df["passenger_count"] > 0]
    log.info(f"After coords/passengers filtering: {len(df):,} rows")

    # 3. Features + KMeans
    preprocessor = TripPreprocessor(n_clusters=settings.KMEANS_CLUSTERS)
    df = preprocessor.transform(df, fit_kmeans=True)

    feature_cols = preprocessor.get_feature_columns()
    X = df[feature_cols]

    # 4. Log target
    y = np.log1p(df["trip_duration"])

    # 5. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.TEST_SIZE,       # ✅ TEST_SIZE=0.2
        random_state=settings.RANDOM_STATE
    )
    log.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}, features: {len(feature_cols)}")

    # 6. CatBoost
    log.info("Training CatBoostRegressor (Optuna params)...")
    model = CatBoostRegressor(
        iterations=619,
        learning_rate=0.145532,
        depth=6,
        l2_leaf_reg=0.153649,
        random_strength=3.905042,
        bagging_temperature=0.493611,
        border_count=191,
        verbose=False,
        random_state=settings.RANDOM_STATE,
        thread_count=-1,
        task_type="CPU",
    )

    log.info("Fitting model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    log.info("Model trained ✓")

    # 7. Metrics
    y_train_pred = np.expm1(model.predict(X_train))
    y_test_pred  = np.expm1(model.predict(X_test))
    y_train_real = np.expm1(y_train)
    y_test_real  = np.expm1(y_test)

    train_rmse  = np.sqrt(mean_squared_error(y_train_real, y_train_pred))
    test_rmse   = np.sqrt(mean_squared_error(y_test_real, y_test_pred))
    test_rmsle  = np.sqrt(mean_squared_log_error(y_test_real, y_test_pred))
    test_mae    = mean_absolute_error(y_test_real, y_test_pred)
    test_r2     = r2_score(y_test_real, y_test_pred)

    metrics = {
        "rmsle":       test_rmsle,
        "rmse":        test_rmse,
        "mae":         test_mae,
        "r2":          test_r2,
        "train_rmse":  train_rmse,
        "overfitting": abs(train_rmse - test_rmse),
    }

    log.info("MODEL METRICS:")
    log.info(f"  RMSLE: {test_rmsle:.6f}")
    log.info(f"  RMSE:  {test_rmse:.2f}")
    log.info(f"  MAE:   {test_mae:.2f}")
    log.info(f"  R²:    {test_r2:.4f}")

    # 8. ✅ Validation БЛОКИРУЕТ деплой при провале
    is_valid = validate_model(metrics)

    if not is_valid:
        log.error(
            f"❌ VALIDATION FAILED! "
            f"RMSLE {metrics['rmsle']:.6f} > threshold {settings.MIN_RMSLE_THRESHOLD}"
        )
        log.error("🚫 Deployment BLOCKED — модель не прошла валидацию")
        return  # ← выходим ДО deploy_model!

    # 9. ✅ Deploy только если валидация прошла
    log.info("✅ Validation passed — deploying model...")
    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    deploy_model(
        model=model,
        preprocessor=preprocessor,
        metrics=metrics,
        version=version,
    )

    log.info(f"TRAINING COMPLETE - Model {version} deployed ✓")


if __name__ == "__main__":
    train_model()

