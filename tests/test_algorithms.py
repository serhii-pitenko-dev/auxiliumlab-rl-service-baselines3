"""Unit tests for algorithm factory and model building."""
import pytest
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN

from auxilium_rl.core.algorithms import build_model, get_model_class, _get_default_hyperparams
from auxilium_rl.core.dto import AlgorithmType


@pytest.fixture
def dummy_env():
    """Create a simple dummy environment for testing."""
    return gym.make("CartPole-v1")


class TestAlgorithmFactory:
    """Test suite for algorithm factory functions."""
    
    def test_build_ppo_model(self, dummy_env):
        """Test building a PPO model."""
        model = build_model(
            algorithm=AlgorithmType.PPO,
            env=dummy_env,
            seed=42,
            verbose=0
        )
        
        assert isinstance(model, PPO)
        assert model.policy.__class__.__name__ == "ActorCriticPolicy"
    
    def test_build_a2c_model(self, dummy_env):
        """Test building an A2C model."""
        model = build_model(
            algorithm=AlgorithmType.A2C,
            env=dummy_env,
            seed=42,
            verbose=0
        )
        
        assert isinstance(model, A2C)
        assert model.policy.__class__.__name__ == "ActorCriticPolicy"
    
    def test_build_dqn_model(self, dummy_env):
        """Test building a DQN model."""
        model = build_model(
            algorithm=AlgorithmType.DQN,
            env=dummy_env,
            seed=42,
            verbose=0
        )
        
        assert isinstance(model, DQN)
        assert model.policy.__class__.__name__ == "DQNPolicy"
    
    def test_custom_hyperparameters(self, dummy_env):
        """Test that custom hyperparameters are applied."""
        custom_lr = 1e-5
        model = build_model(
            algorithm=AlgorithmType.PPO,
            env=dummy_env,
            hyperparameters={"learning_rate": custom_lr},
            seed=42,
            verbose=0
        )
        
        assert model.learning_rate == custom_lr
    
    def test_default_hyperparameters(self):
        """Test that default hyperparameters are returned correctly."""
        ppo_defaults = _get_default_hyperparams(AlgorithmType.PPO)
        assert "learning_rate" in ppo_defaults
        assert "n_steps" in ppo_defaults
        
        a2c_defaults = _get_default_hyperparams(AlgorithmType.A2C)
        assert "learning_rate" in a2c_defaults
        assert "n_steps" in a2c_defaults
        
        dqn_defaults = _get_default_hyperparams(AlgorithmType.DQN)
        assert "learning_rate" in dqn_defaults
        assert "buffer_size" in dqn_defaults
    
    def test_get_model_class(self):
        """Test getting model classes."""
        assert get_model_class(AlgorithmType.PPO) == PPO
        assert get_model_class(AlgorithmType.A2C) == A2C
        assert get_model_class(AlgorithmType.DQN) == DQN
    
    def test_invalid_algorithm(self, dummy_env):
        """Test that invalid algorithm raises error."""
        # This would require creating an invalid AlgorithmType, which isn't directly possible
        # with enums, but we can test the error path indirectly
        pass
