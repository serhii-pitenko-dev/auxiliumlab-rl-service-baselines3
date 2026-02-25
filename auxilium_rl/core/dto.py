"""Data Transfer Objects for internal use."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class AlgorithmType(Enum):
    """Supported RL algorithms."""
    PPO = "ppo"
    A2C = "a2c"
    DQN = "dqn"


class RunStatus(Enum):
    """Training run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    algorithm: AlgorithmType
    experiment_id: str
    total_timesteps: int
    seed: int
    hyperparameters: Dict[str, str] = field(default_factory=dict)
    model_output_path: str = ""
    
    def get_hyperparams_typed(self) -> Dict[str, Any]:
        """Convert string hyperparameters to appropriate types."""
        typed_params = {}
        
        # Common numeric hyperparameters
        float_params = {"learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "clip_range"}
        int_params = {"n_steps", "batch_size", "n_epochs", "buffer_size", "learning_starts"}
        
        for key, value in self.hyperparameters.items():
            if key in float_params:
                typed_params[key] = float(value)
            elif key in int_params:
                typed_params[key] = int(value)
            else:
                # Keep as string or try to infer
                try:
                    typed_params[key] = float(value)
                except ValueError:
                    typed_params[key] = value
        
        return typed_params


@dataclass
class RunInfo:
    """Information about a training run."""
    run_id: str
    config: TrainingConfig
    status: RunStatus
    timesteps_done: int = 0
    total_timesteps: int = 0
    last_checkpoint_path: Optional[str] = None
    final_model_path: Optional[str] = None
    error_message: Optional[str] = None
    model_in_memory: Optional[Any] = None  # For inference after training
