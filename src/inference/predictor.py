# src/inference/predictor.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.schemas import TripRequest, TripResponse
from src.common.preprocessing import TripPreprocessor


class TripPredictor:
    def __init__(
        self,
        model,
        preprocessor: TripPreprocessor,
        version: str,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.version = version

    def predict(self, trip: TripRequest) -> TripResponse:
        # 1. TripRequest -> DataFrame
        df = pd.DataFrame([trip.model_dump()])

        # 2. Feature engineering (fit_kmeans=False, используем уже обученные модели)
        df_features = self.preprocessor.transform(df, fit_kmeans=False)
        feature_cols = self.preprocessor.get_feature_columns()
        X = df_features[feature_cols]

        # 3. Предсказание лог-таргета
        y_log = float(self.model.predict(X)[0])

        # 4. Обратное преобразование в секунды
        y_seconds = float(np.expm1(y_log))
        y_minutes = y_seconds / 60.0

        return TripResponse(
            predicted_duration_seconds=y_seconds,
            predicted_duration_minutes=y_minutes,
            model_version=self.version,
        )