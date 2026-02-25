"""Gymnasium environment wrapper for external simulation."""
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..infra.external_env_adapter import ExternalEnvAdapter

logger = logging.getLogger(__name__)


class ExternalSimEnv(gym.Env):
    """
    Gymnasium environment that wraps an external simulation via adapter.
    
    This environment delegates reset/step calls to an external adapter,
    which can communicate with a .NET simulation or use a fake implementation.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        adapter: ExternalEnvAdapter,
        observation_dim: int = 4,
        action_dim: int = 4,
        max_steps: int = 500,
    ):
        """
        Initialize the environment.
        
        Args:
            adapter: External environment adapter
            observation_dim: Dimensionality of observation space
            action_dim: Number of discrete actions
            max_steps: Maximum steps before truncation
        """
        super().__init__()
        
        self.adapter = adapter
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
        
        self._current_step = 0
        self._last_obs = None
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset via adapter
        observation = self.adapter.reset(seed=seed)
        
        # Ensure correct shape and type
        observation = np.array(observation, dtype=np.float32)
        if observation.shape[0] != self.observation_dim:
            logger.warning(
                f"Observation dimension mismatch: expected {self.observation_dim}, "
                f"got {observation.shape[0]}. Padding/truncating."
            )
            observation = self._fix_observation_shape(observation)
        
        self._current_step = 0
        self._last_obs = observation
        
        info = {}
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Delegate to adapter
        observation, reward, terminated, truncated, info = self.adapter.step(action)
        
        # Ensure correct shape and type
        observation = np.array(observation, dtype=np.float32)
        if observation.shape[0] != self.observation_dim:
            observation = self._fix_observation_shape(observation)
        
        self._current_step += 1
        self._last_obs = observation
        
        # Apply max_steps truncation
        if self._current_step >= self.max_steps:
            truncated = True
            info["TimeLimit.truncated"] = True
        
        return observation, float(reward), bool(terminated), bool(truncated), info
    
    def close(self) -> None:
        """Clean up resources."""
        self.adapter.close()
    
    def _fix_observation_shape(self, obs: np.ndarray) -> np.ndarray:
        """
        Fix observation shape to match expected dimension.
        
        Args:
            obs: Observation array
            
        Returns:
            Fixed observation
        """
        if obs.shape[0] < self.observation_dim:
            # Pad with zeros
            padded = np.zeros(self.observation_dim, dtype=np.float32)
            padded[:obs.shape[0]] = obs
            return padded
        else:
            # Truncate
            return obs[:self.observation_dim]
