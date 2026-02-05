#!/usr/bin/env python3
"""
FINAL FIXED: Preprocessor datetime OK + post-clean numeric features
"""

import argparse
import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

try:
    from inference.model_loader import load_active_model
    from common.logger import log
    print("✅ Imports OK")
except ImportError as e:
    print(f"❌ {e}")
    sys.exit(1)

def validate_columns(df):
    required = ['pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    missing = [col for col in required if col not in df.columns]
    if missing: raise ValueError(f"Missing: {missing}")
    print(f"✅ Columns: {len(df.columns)}")

def predict_new_data(input_file: str, output_file: str = None):
    log.info("🚀 Batch predict")
    
    model, preprocessor, metadata = load_active_model()
    version = metadata['version']
    rmsle = metadata.get('metrics', {}).get('rmsle', 'N/A')
    log.info(f"✅ v{version} RMSLE: {rmsle}")
    
    df = pd.read_csv(input_file)
    log.info(f"📊 {len(df):,} rows")
    validate_columns(df)
    
    # Drop junk
    df = df.drop(columns=['id', 'trip_duration'], errors='ignore')
    
    # **Datetime prep (KEEP raw для preprocessor!)**
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
        df['pickup_hour'] = df['pickup_datetime'].dt.hour.astype(int)
        df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek.astype(int)
        df['pickup_month'] = df['pickup_datetime'].dt.month.astype(int)
        log.info("⏰ Features extracted (raw KEPT)")
    
    # Categoricals → numeric
    for col in ['vendor_id', 'passenger_count', 'store_and_fwd_flag']:
        if col in df: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    log.info(f"✅ Clean: {df.shape}")
    
    # **Preprocessor (uses pickup_datetime internally)**
    features = preprocessor.transform(df)
    log.info(f"📊 Raw features: {features.shape}")
    
    # **POST-FIX: Remove datetime + FORCE numeric**
    # Drop any datetime/object columns
    datetime_cols = features.select_dtypes(include=['datetime', 'object']).columns
    if len(datetime_cols) > 0:
        log.info(f"🧹 Dropping datetime cols: {list(datetime_cols)}")
        features = features.drop(columns=datetime_cols, errors='ignore')
    
    # Cast ONLY numeric columns to float64
    numeric_features = features.select_dtypes(include=[np.number])
    if numeric_features.shape[1] == 0:
        raise ValueError("No numeric features!")
    
    features = numeric_features.astype('float64')
    
    log.info(f"✅ Numeric features: ({features.shape[0]:,}, {features.shape[1]})")
    log.info(f"🔍 Sample dtypes: {features.dtypes[:5].to_dict()}")
    
    # **Predict**
    log.info("🤖 CatBoost...")
    log_preds = model.predict(features)
    predictions = np.expm1(log_preds)
    
    # **Results**
    df_result = df.copy()
    df_result['predicted_duration_seconds'] = predictions
    df_result['predicted_duration_minutes'] = predictions / 60
    df_result['model_version'] = version
    
    mean_pred = predictions.mean()
    p95 = np.percentile(predictions, 95)
    log.info(f"📈 Mean: {mean_pred:.1f}s | P95: {p95:.1f}s")
    
    # **Save**
    output_path = Path(output_file or 'data/preds.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)
    log.info(f"💾 {len(df_result):,} rows → {output_path}")
    print("🎉 SUCCESS!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYC Taxi Predict")
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data/preds.csv')
    args = parser.parse_args()
    
    try:
        predict_new_data(args.input, args.output)
    except Exception as e:
        log.error(f"💥 {e}")
        traceback.print_exc()
        sys.exit(1)



