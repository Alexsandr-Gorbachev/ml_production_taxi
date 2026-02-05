# src/inference/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Устанавливаем uv
RUN pip install --no-cache-dir uv

# Копируем зависимости
COPY pyproject.toml .

# Ставим зависимости через uv
RUN uv pip install --system -e .

# Копируем исходники
COPY src/ ./src/

# Экспортируем порт
EXPOSE 8000

# Запускаем приложение
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
