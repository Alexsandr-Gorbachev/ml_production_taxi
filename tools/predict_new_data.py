#!/usr/bin/env python3
"""
Batch prediction tool for NYC Taxi duration.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.inference.model_loader import load_active_model  # ✅ абсолютный импорт
from src.common.logger import log                         # ✅ абсолютный импорт


def validate_columns(df: pd.DataFrame) -> None:
    required = [
        'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count'
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    log.info(f"✅ Columns OK ({len(df.columns)} total)")  # ✅ log вместо print


def predict_new_data(input_file: str, output_file: str = None) -> None:
    log.info("🚀 Starting batch prediction")

    # 1. Загрузка модели
    model, preprocessor, metadata = load_active_model()
    version = metadata['version']
    rmsle = metadata.get('metrics', {}).get('rmsle', 'N/A')
    log.info(f"✅ Loaded model v{version}, RMSLE: {rmsle}")

    # 2. Загрузка данных
    df = pd.read_csv(input_file)
    log.info(f"📊 Loaded {len(df):,} rows")
    validate_columns(df)

    # 3. Убираем лишние колонки
    df = df.drop(columns=['id', 'trip_duration'], errors='ignore')

    # 4. ✅ ТОЛЬКО конвертируем тип — фичи создаёт preprocessor сам!
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        log.info("⏰ pickup_datetime converted to datetime")
        # ✅ НЕ создаём pickup_hour/dayofweek/month здесь — это делает transform()

    # 5. Числовые колонки
    for col in ['vendor_id', 'passenger_count', 'store_and_fwd_flag']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    log.info(f"✅ Data prepared: {df.shape}")

    # 6. Preprocessing
    features = preprocessor.transform(df, fit_kmeans=False)
    log.info(f"📊 After transform: {features.shape}")

    # 7. ✅ СТРОГО нужные фичи в нужном порядке
    feature_cols = preprocessor.get_feature_columns()
    features = features[feature_cols]
    log.info(f"✅ Final features: {features.shape}")

    # 8. Предсказание
    log.info("🤖 Running CatBoost predictions...")
    log_preds = model.predict(features)
    predictions = np.expm1(log_preds)

    # 9. Результаты
    df_result = df.copy()
    df_result['predicted_duration_seconds'] = predictions
    df_result['predicted_duration_minutes'] = predictions / 60
    df_result['model_version'] = version

    mean_pred = predictions.mean()
    p95 = np.percentile(predictions, 95)
    log.info(f"📈 Mean: {mean_pred:.1f}s ({mean_pred/60:.1f} min) | P95: {p95:.1f}s")

    # 10. Сохранение
    output_path = Path(output_file or 'data/preds.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)
    log.info(f"💾 Saved {len(df_result):,} rows → {output_path}")
    log.info("🎉 Batch prediction complete!")  # ✅ log вместо print


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Batch Predictor")
    parser.add_argument('--input', required=True, help="Path to input CSV")
    parser.add_argument('--output', default='data/preds.csv', help="Path to output CSV")
    args = parser.parse_args()

    try:
        predict_new_data(args.input, args.output)
    except Exception as e:
        log.error(f"💥 Prediction failed: {e}")
        sys.exit(1)



