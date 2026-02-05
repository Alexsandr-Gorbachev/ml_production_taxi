FROM python:3.10-slim

WORKDIR /app

# Установка uv
RUN pip install --no-cache-dir uv

# Копируем зависимости
COPY pyproject.toml ./

# Устанавливаем зависимости
RUN uv pip install --system -e .

# Копируем исходники
COPY src/ ./src/

# Запускаем обучение
CMD ["python", "-m", "src.training.train"]
