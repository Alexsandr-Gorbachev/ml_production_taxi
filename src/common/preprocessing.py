# src/common/preprocessing.py
"""
Production preprocessing utilities for NYC Taxi ML service.

Перенос логики из helpers.py (ноутбук 04) в прод-формат:
- расстояния (Haversine, Manhattan)
- временные признаки
- географические признаки
- KMeans-кластеры pickup/dropoff
- взаимодействия и флаги
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Векторизованный Haversine distance в километрах."""
    lat1, lon1, lat2, lon2 = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


def manhattan_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Векторизованный Manhattan distance по поверхности Земли (км)."""
    lat_distance = np.abs(lat2 - lat1) * 111
    lon_distance = (
        np.abs(lon2 - lon1)
        * 111
        * np.cos(np.radians((lat1 + lat2) / 2))
    )
    return lat_distance + lon_distance


@dataclass
class TripPreprocessor:
    """
    Препроцессор для признаков, согласованный с ноутбуком 04 + helpers.py.

    - fit_kmeans=True (Training): обучаем kmeans на pickup/dropoff координатах
    - fit_kmeans=False (Inference): используем уже обученные модели
    """

    n_clusters: int = 10
    kmeans_pickup: Optional[KMeans] = None
    kmeans_dropoff: Optional[KMeans] = None

    # фиксированный порядок признаков для модели
    def get_feature_columns(self) -> List[str]:
        return [
            # расстояния
            "haversine_distance",
            "manhattan_distance",
            "haversine_distance_squared",
            "manhattan_distance_squared",
            "distance_ratio",
            # флаги
            "is_zero_distance",
            "is_very_short_trip",
            # временные
            "hour",
            "day_of_week",
            "month",
            "day_of_month",
            "minute",
            "time_period",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            # географические округленные координаты
            "pickup_lat_rounded",
            "pickup_lon_rounded",
            "dropoff_lat_rounded",
            "dropoff_lon_rounded",
            # KMeans кластеры
            "pickup_cluster",
            "dropoff_cluster",
            "same_cluster",
            # взаимодействия
            "distance_hour_interaction",
            "distance_dow_interaction",
            # базовые
            "passenger_count",
            "vendor_id",
        ]

    # внутренний метод: обучаем KMeans
    def _fit_kmeans(self, df: pd.DataFrame) -> None:
        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        self.kmeans_pickup = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        ).fit(pickup_coords)

        self.kmeans_dropoff = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        ).fit(dropoff_coords)

    def transform(self, df: pd.DataFrame, fit_kmeans: bool = False) -> pd.DataFrame:
        """
        Применяет полный feature engineering.

        - df: DataFrame со столбцами:
          ['pickup_datetime', 'pickup_latitude', 'pickup_longitude',
           'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'vendor_id']
        - fit_kmeans=True: обучает KMeans (используется в Training)
        - fit_kmeans=False: ожидает, что self.kmeans_* уже заданы (Inference)
        """
        df = df.copy()

        # 1. Datetime
        if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
            df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        df["hour"] = df["pickup_datetime"].dt.hour
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
        df["month"] = df["pickup_datetime"].dt.month
        df["day_of_month"] = df["pickup_datetime"].dt.day
        df["minute"] = df["pickup_datetime"].dt.minute

        def time_period_fn(h: int) -> int:
            if 6 <= h < 12:
                return 0  # утро
            if 12 <= h < 17:
                return 1  # день
            if 17 <= h < 22:
                return 2  # вечер
            return 3      # ночь

        df["time_period"] = df["hour"].apply(time_period_fn)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # 2. Расстояния
        df["haversine_distance"] = haversine_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        df["manhattan_distance"] = manhattan_distance(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
        df["haversine_distance_squared"] = df["haversine_distance"] ** 2
        df["manhattan_distance_squared"] = df["manhattan_distance"] ** 2
        df["distance_ratio"] = (df["manhattan_distance"] + 1e-5) / (
            df["haversine_distance"] + 1e-5
        )

        df["is_zero_distance"] = (df["haversine_distance"] == 0).astype(int)
        df["is_very_short_trip"] = (df["haversine_distance"] < 0.1).astype(int)

        # 3. География (округлённые координаты)
        df["pickup_lat_rounded"] = (df["pickup_latitude"] * 10).astype(int) / 10
        df["pickup_lon_rounded"] = (df["pickup_longitude"] * 10).astype(int) / 10
        df["dropoff_lat_rounded"] = (df["dropoff_latitude"] * 10).astype(int) / 10
        df["dropoff_lon_rounded"] = (df["dropoff_longitude"] * 10).astype(int) / 10

        # 4. KMeans
        if fit_kmeans:
            self._fit_kmeans(df)

        if self.kmeans_pickup is None or self.kmeans_dropoff is None:
            raise ValueError(
                "KMeans models are not set. "
                "Use fit_kmeans=True in training or load models in inference."
            )

        pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
        dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

        df["pickup_cluster"] = self.kmeans_pickup.predict(pickup_coords)
        df["dropoff_cluster"] = self.kmeans_dropoff.predict(dropoff_coords)
        df["same_cluster"] = (
            df["pickup_cluster"] == df["dropoff_cluster"]
        ).astype(int)

        # 5. Взаимодействия
        df["distance_hour_interaction"] = df["haversine_distance"] * df["hour"]
        df["distance_dow_interaction"] = (
            df["haversine_distance"] * df["day_of_week"]
        )

        return df