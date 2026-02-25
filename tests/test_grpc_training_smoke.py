"""Smoke test for gRPC training endpoints."""
import pytest
import grpc
import time
import numpy as np
from pathlib import Path

from generated import policy_trainer_pb2, policy_trainer_pb2_grpc
from auxilium_rl.infra.config import ServiceConfig, EnvConfig
from auxilium_rl.infra.model_store import ModelStore
from auxilium_rl.core.training import TrainingOrchestrator
from auxilium_rl.transport.grpc_server import create_server


@pytest.fixture(scope="module")
def test_server():
    """
    Create and start a test gRPC server on a random port.
    Yields the server and port, then stops the server after tests.
    """
    # Use a fixed port for testing to avoid port binding issues
    test_port = 50052
    
    # Use a test-specific configuration
    config = ServiceConfig(
        host="localhost",
        port=test_port,
        max_workers=4,
        checkpoint_dir="./test_checkpoints",
        models_dir="./test_models"
    )
    
    env_config = EnvConfig(
        observation_dim=4,
        action_dim=4,
        max_steps=100
    )
    
    # Clean up test directories before starting
    import shutil
    for dir_path in [config.checkpoint_dir, config.models_dir]:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
    
    # Create orchestrator and server
    model_store = ModelStore(
        models_dir=config.models_dir,
        checkpoint_dir=config.checkpoint_dir
    )
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=1000
    )
    
    server = create_server(orchestrator, config)
    server.start()
    
    yield test_port
    
    # Cleanup
    server.stop()
    
    # Clean up test directories
    for dir_path in [config.checkpoint_dir, config.models_dir]:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)


@pytest.fixture
def grpc_stub(test_server):
    """Create a gRPC stub for testing."""
    channel = grpc.insecure_channel(f"localhost:{test_server}")
    stub = policy_trainer_pb2_grpc.PolicyTrainerServiceStub(channel)
    yield stub
    channel.close()


class TestGrpcTrainingSmoke:
    """Smoke tests for gRPC training endpoints."""
    
    def test_start_training_ppo(self, grpc_stub):
        """Test starting PPO training and checking status."""
        # Start training with minimal timesteps
        request = policy_trainer_pb2.TrainingRequest(
            experiment_id="test_ppo_smoke",
            total_timesteps=256,  # Small number for quick test
            seed=42,
            hyperparameters={
                "learning_rate": "3e-4",
                "n_steps": "64",
                "batch_size": "32"
            },
            model_output_path=""
        )
        
        response = grpc_stub.StartTrainingPPO(request)
        
        assert response.status == policy_trainer_pb2.STARTED
        assert response.run_id != ""
        assert "successfully" in response.message.lower()
        
        run_id = response.run_id
        
        # Poll status until training completes
        max_attempts = 30
        for _ in range(max_attempts):
            status_request = policy_trainer_pb2.StatusRequest(run_id=run_id)
            status_response = grpc_stub.GetTrainingStatus(status_request)
            
            if status_response.is_done:
                assert status_response.error_message == ""
                assert status_response.timesteps_done == 256
                break
            
            time.sleep(1)
        else:
            pytest.fail("Training did not complete within timeout")
        
        # Verify model file exists (check the run registry for path)
        # Since we can't easily access the path, we'll just verify inference works
    
    def test_start_training_a2c(self, grpc_stub):
        """Test starting A2C training."""
        request = policy_trainer_pb2.TrainingRequest(
            experiment_id="test_a2c_smoke",
            total_timesteps=256,
            seed=42,
            hyperparameters={},
            model_output_path=""
        )
        
        response = grpc_stub.StartTrainingA2C(request)
        
        assert response.status == policy_trainer_pb2.STARTED
        assert response.run_id != ""
    
    def test_start_training_dqn(self, grpc_stub):
        """Test starting DQN training."""
        request = policy_trainer_pb2.TrainingRequest(
            experiment_id="test_dqn_smoke",
            total_timesteps=256,
            seed=42,
            hyperparameters={},
            model_output_path=""
        )
        
        response = grpc_stub.StartTrainingDQN(request)
        
        assert response.status == policy_trainer_pb2.STARTED
        assert response.run_id != ""
    
    def test_inference_after_training(self, grpc_stub):
        """Test inference endpoint after training completes."""
        # Start training
        request = policy_trainer_pb2.TrainingRequest(
            experiment_id="test_inference",
            total_timesteps=256,
            seed=42,
            hyperparameters={},
            model_output_path=""
        )
        
        response = grpc_stub.StartTrainingPPO(request)
        run_id = response.run_id
        
        # Wait for training to complete
        max_attempts = 30
        for _ in range(max_attempts):
            status_request = policy_trainer_pb2.StatusRequest(run_id=run_id)
            status_response = grpc_stub.GetTrainingStatus(status_request)
            
            if status_response.is_done:
                break
            time.sleep(1)
        
        # Perform inference
        observation = [0.1, 0.2, 0.3, 0.4]
        act_request = policy_trainer_pb2.ActRequest(
            run_id=run_id,
            observation=observation
        )
        
        act_response = grpc_stub.Act(act_request)
        
        assert act_response.success
        assert 0 <= act_response.action < 4  # Valid action for our env
        assert act_response.error_message == ""
    
    def test_invalid_run_id_status(self, grpc_stub):
        """Test querying status for non-existent run."""
        status_request = policy_trainer_pb2.StatusRequest(run_id="invalid-run-id")
        status_response = grpc_stub.GetTrainingStatus(status_request)
        
        assert not status_response.is_done
        assert "not found" in status_response.error_message.lower()
    
    def test_invalid_run_id_inference(self, grpc_stub):
        """Test inference with non-existent run."""
        act_request = policy_trainer_pb2.ActRequest(
            run_id="invalid-run-id",
            observation=[0.1, 0.2, 0.3, 0.4]
        )
        
        act_response = grpc_stub.Act(act_request)
        
        assert not act_response.success
        assert act_response.error_message != ""
