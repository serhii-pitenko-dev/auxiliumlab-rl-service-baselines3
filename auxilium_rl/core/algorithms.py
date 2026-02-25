"""RL algorithm factory and model builders."""
import logging
from typing import Dict, Any, Optional
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import gymnasium as gym

from .dto import AlgorithmType

logger = logging.getLogger(__name__)


def build_model(
    algorithm: AlgorithmType,
    env: gym.Env,
    hyperparameters: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    verbose: int = 1
) -> BaseAlgorithm:
    """
    Build a Stable Baselines3 model.
    
    Args:
        algorithm: Type of algorithm to build
        env: Gymnasium environment
        hyperparameters: Optional hyperparameter overrides
        seed: Random seed
        verbose: Verbosity level
        
    Returns:
        Initialized SB3 model
    """
    hyperparameters = hyperparameters or {}
    
    # Default hyperparameters for each algorithm
    defaults = _get_default_hyperparams(algorithm)
    
    # Merge with provided hyperparameters
    params = {**defaults, **hyperparameters}
    
    logger.info(f"Building {algorithm.value.upper()} model with params: {params}")
    
    if algorithm == AlgorithmType.PPO:
        return PPO(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=verbose,
            **params
        )
    elif algorithm == AlgorithmType.A2C:
        return A2C(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=verbose,
            **params
        )
    elif algorithm == AlgorithmType.DQN:
        return DQN(
            policy="MlpPolicy",
            env=env,
            seed=seed,
            verbose=verbose,
            **params
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def _get_default_hyperparams(algorithm: AlgorithmType) -> Dict[str, Any]:
    """Get default hyperparameters for an algorithm."""
    
    if algorithm == AlgorithmType.PPO:
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
        }
    elif algorithm == AlgorithmType.A2C:
        return {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
        }
    elif algorithm == AlgorithmType.DQN:
        return {
            "learning_rate": 1e-4,
            "buffer_size": 50000,
            "learning_starts": 1000,
            "batch_size": 32,
            "gamma": 0.99,
            "train_freq": 4,
            "target_update_interval": 1000,
        }
    else:
        return {}


def get_model_class(algorithm: AlgorithmType) -> type:
    """
    Get the SB3 model class for an algorithm.
    
    Args:
        algorithm: Algorithm type
        
    Returns:
        Model class
    """
    if algorithm == AlgorithmType.PPO:
        return PPO
    elif algorithm == AlgorithmType.A2C:
        return A2C
    elif algorithm == AlgorithmType.DQN:
        return DQN
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
