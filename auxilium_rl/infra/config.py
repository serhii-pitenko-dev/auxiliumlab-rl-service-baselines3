"""Configuration management for the RL training service."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ServiceConfig:
    """Configuration for the gRPC service."""
    
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    checkpoint_dir: str = "./checkpoints"
    models_dir: str = "./trained_models"
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("GRPC_HOST", "0.0.0.0"),
            port=int(os.getenv("GRPC_PORT", "50051")),
            max_workers=int(os.getenv("GRPC_MAX_WORKERS", "10")),
            checkpoint_dir=os.getenv("CHECKPOINT_DIR", "./checkpoints"),
            models_dir=os.getenv("MODELS_DIR", "./trained_models"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


@dataclass
class EnvConfig:
    """Configuration for the Gymnasium environment."""
    
    observation_dim: int = 4
    action_dim: int = 4
    max_steps: int = 500
    
    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Load environment configuration from environment variables."""
        return cls(
            observation_dim=int(os.getenv("OBS_DIM", "4")),
            action_dim=int(os.getenv("ACTION_DIM", "4")),
            max_steps=int(os.getenv("MAX_STEPS", "500")),
        )
