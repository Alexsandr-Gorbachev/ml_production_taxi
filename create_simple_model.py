import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import joblib

print("🔄 Создаем ПРОСТУЮ модель для Этапа 1...")

# Простые фичи (только координаты + расстояние)
X = pd.DataFrame({
    'haversine_distance': np.random.uniform(1, 10, 100),
    'pickup_longitude': np.random.uniform(-74, -73.9, 100),
    'pickup_latitude': np.random.uniform(40.7, 40.8, 100),
    'dropoff_longitude': np.random.uniform(-74, -73.9, 100),
    'dropoff_latitude': np.random.uniform(40.7, 40.8, 100),
    'passenger_count': np.random.randint(1, 5, 100)
})
y = np.random.uniform(300, 1800, 100)

model = CatBoostRegressor(iterations=50, depth=3, verbose=0)
model.fit(X, y)

joblib.dump(model, 'models/model.pkl')
print('✅ Простая модель создана! Фичи:', list(X.columns))
