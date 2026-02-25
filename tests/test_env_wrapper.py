"""Unit tests for the Gymnasium environment wrapper."""
import pytest
import numpy as np

from auxilium_rl.core.env import ExternalSimEnv
from auxilium_rl.infra.external_env_adapter import FakeExternalEnvAdapter


@pytest.fixture
def fake_adapter():
    """Create a fake external environment adapter."""
    return FakeExternalEnvAdapter(observation_dim=4, action_dim=4)


@pytest.fixture
def env_wrapper(fake_adapter):
    """Create an environment wrapper with fake adapter."""
    return ExternalSimEnv(
        adapter=fake_adapter,
        observation_dim=4,
        action_dim=4,
        max_steps=100
    )


class TestEnvironmentWrapper:
    """Test suite for the environment wrapper."""
    
    def test_initialization(self, env_wrapper):
        """Test that environment initializes correctly."""
        assert env_wrapper.observation_space.shape == (4,)
        assert env_wrapper.action_space.n == 4
        assert env_wrapper.max_steps == 100
    
    def test_reset(self, env_wrapper):
        """Test environment reset."""
        observation, info = env_wrapper.reset(seed=42)
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (4,)
        assert observation.dtype == np.float32
        assert isinstance(info, dict)
    
    def test_step(self, env_wrapper):
        """Test environment step."""
        env_wrapper.reset(seed=42)
        observation, reward, terminated, truncated, info = env_wrapper.step(2)
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (4,)
        assert observation.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_max_steps_truncation(self, env_wrapper):
        """Test that environment truncates after max_steps."""
        env_wrapper.reset(seed=42)
        
        for i in range(env_wrapper.max_steps - 1):
            _, _, terminated, truncated, _ = env_wrapper.step(0)
            if terminated:
                break  # Episode ended naturally
        
        # Take one more step to reach max_steps
        if not terminated:
            _, _, _, truncated, info = env_wrapper.step(0)
            assert truncated
            assert "TimeLimit.truncated" in info
    
    def test_deterministic_with_seed(self, fake_adapter):
        """Test that environment is deterministic with same seed."""
        env1 = ExternalSimEnv(fake_adapter, observation_dim=4, action_dim=4)
        env2 = ExternalSimEnv(
            FakeExternalEnvAdapter(observation_dim=4, action_dim=4),
            observation_dim=4,
            action_dim=4
        )
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        np.testing.assert_array_almost_equal(obs1, obs2)
        
        # Take same actions
        for _ in range(10):
            obs1, _, term1, trunc1, _ = env1.step(2)
            obs2, _, term2, trunc2, _ = env2.step(2)
            
            if term1 or trunc1:
                break
            
            np.testing.assert_array_almost_equal(obs1, obs2)
    
    def test_adapter_wiring(self, env_wrapper):
        """Test that environment correctly calls adapter methods."""
        # Reset should call adapter.reset()
        obs, _ = env_wrapper.reset(seed=42)
        assert obs is not None
        
        # Step should call adapter.step()
        obs, reward, terminated, truncated, info = env_wrapper.step(1)
        assert obs is not None
        assert isinstance(reward, float)
    
    def test_observation_shape_fixing(self, fake_adapter):
        """Test that mismatched observation shapes are corrected."""
        # Create environment expecting different dimensions
        env = ExternalSimEnv(
            adapter=fake_adapter,
            observation_dim=8,  # Adapter provides 4, env expects 8
            action_dim=4,
            max_steps=100
        )
        
        obs, _ = env.reset(seed=42)
        assert obs.shape == (8,)
        
        # First 4 should be from adapter, rest should be padded zeros
        assert not np.allclose(obs[:4], 0)  # Adapter values
