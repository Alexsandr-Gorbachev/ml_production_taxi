"""
Structured logging for ML service (NYC Taxi MVP)
Compatible with Этап 3-4: глобальный `log` + service_name
"""
from loguru import logger
import sys
from pathlib import Path

def setup_logger(service_name: str = "ml-service"):
    """
    Setup structured logger with rotation and service name prefix
    
    Args:
        service_name: "inference", "training", "ml-service"
    """
    # Remove all handlers
    logger.remove()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Console sink (красиво в Docker logs)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               f"<blue>{service_name}</blue> | "
               "<level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        diagnose=True
    )
    
    # File sink с ротацией
    logger.add(
        logs_dir / f"{service_name}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {name} | {level} | {message}",
        diagnose=True
    )
    
    logger.info(f"🚀 Logger initialized for {service_name}")
    return logger

# Глобальный log для обратной совместимости (train.py, app.py и т.д.)
log = logger

# Автоинициализация при импорте
setup_logger("ml-service")