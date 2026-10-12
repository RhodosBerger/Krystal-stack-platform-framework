"""
Krystal-Stack Core Python Package
"""

from src.python.krystal_sdk import (
    Krystal,
    create_game_optimizer,
    create_iot_optimizer,
    create_ml_optimizer,
    create_server_optimizer,
    health_check,
)

KrystalSDK = Krystal
create_gpu_optimizer = create_game_optimizer

__all__ = [
    "Krystal",
    "KrystalSDK",
    "create_game_optimizer",
    "create_gpu_optimizer",
    "create_iot_optimizer",
    "create_ml_optimizer",
    "create_server_optimizer",
    "health_check",
]
