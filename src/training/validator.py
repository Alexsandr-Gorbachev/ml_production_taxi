"""
Model validation logic
"""
from typing import Dict
from src.common.config import settings
from src.common.logger import log


def validate_model(metrics: Dict[str, float]) -> bool:
    """
    Валидация метрик модели перед деплоем
    
    Args:
        metrics: Словарь с метриками модели
        
    Returns:
        True если модель прошла валидацию
    """
    log.info("Validating model metrics...")
    
    # Основная проверка: RMSLE не должен превышать порог
    rmsle = metrics.get('rmsle', float('inf'))
    threshold = settings.MIN_RMSLE_THRESHOLD
    
    if rmsle > threshold:
        log.error(f"❌ RMSLE {rmsle:.6f} exceeds threshold {threshold}")
        return False
    
    # Дополнительные проверки
    r2 = metrics.get('r2', 0)
    if r2 < 0.3:
        log.warning(f"⚠️  R² is low: {r2:.4f}")
    
    overfitting = metrics.get('overfitting', 0)
    if overfitting > 100:
        log.warning(f"⚠️  High overfitting detected: {overfitting:.2f}")
    
    log.info(f"✅ Model validation passed (RMSLE: {rmsle:.6f} < {threshold})")
    return True
