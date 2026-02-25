"""Tests for health check functionality."""
import pytest
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from pathlib import Path

from auxilium_rl.infra.config import ServiceConfig, EnvConfig
from auxilium_rl.infra.model_store import ModelStore
from auxilium_rl.core.training import TrainingOrchestrator
from auxilium_rl.transport.grpc_server import create_server
from auxilium_rl.transport.health_servicer import HealthServicer


def test_health_servicer_serving():
    """Test that health servicer returns SERVING when components are healthy."""
    # Create test dependencies
    model_store = ModelStore(
        models_dir="trained_models",
        checkpoint_dir="checkpoints"
    )
    env_config = EnvConfig()
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=10000
    )
    
    # Create health servicer
    health_servicer = HealthServicer(orchestrator, model_store)
    
    # Make health check request
    request = health_pb2.HealthCheckRequest()
    response = health_servicer.Check(request, None)
    
    # Verify response
    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_health_servicer_not_serving_no_orchestrator():
    """Test that health servicer returns NOT_SERVING when orchestrator is None."""
    model_store = ModelStore(
        models_dir="trained_models",
        checkpoint_dir="checkpoints"
    )
    
    # Create health servicer with None orchestrator
    health_servicer = HealthServicer(None, model_store)  # type: ignore[arg-type]
    
    # Make health check request
    request = health_pb2.HealthCheckRequest()
    response = health_servicer.Check(request, None)
    
    # Verify response
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_health_servicer_not_serving_no_model_store():
    """Test that health servicer returns NOT_SERVING when model store is None."""
    env_config = EnvConfig()
    model_store = ModelStore(
        models_dir="trained_models",
        checkpoint_dir="checkpoints"
    )
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=10000
    )
    
    # Create health servicer with None model store
    health_servicer = HealthServicer(orchestrator, None)  # type: ignore[arg-type]
    
    # Make health check request
    request = health_pb2.HealthCheckRequest()
    response = health_servicer.Check(request, None)
    
    # Verify response
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING


def test_health_servicer_not_serving_invalid_directories(tmp_path):
    """Test that health servicer returns NOT_SERVING when directories don't exist."""
    # Create orchestrator with non-existent directories
    nonexistent_models = str(tmp_path / "nonexistent_models")
    nonexistent_checkpoints = str(tmp_path / "nonexistent_checkpoints")
    
    # Ensure directories don't exist initially
    assert not Path(nonexistent_models).exists()
    assert not Path(nonexistent_checkpoints).exists()
    
    model_store = ModelStore(
        models_dir=nonexistent_models,
        checkpoint_dir=nonexistent_checkpoints
    )
    
    # ModelStore creates directories, so delete them to simulate missing directories
    import shutil
    shutil.rmtree(nonexistent_models)
    shutil.rmtree(nonexistent_checkpoints)
    
    env_config = EnvConfig()
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=10000
    )
    
    # Create health servicer
    health_servicer = HealthServicer(orchestrator, model_store)
    
    # Make health check request
    request = health_pb2.HealthCheckRequest()
    response = health_servicer.Check(request, None)
    
    # Verify response
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING


@pytest.mark.integration
def test_grpc_server_health_check_integration():
    """Integration test for health check through gRPC server."""
    import threading
    import time
    import socket
    
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    
    # Setup
    env_config = EnvConfig.from_env()
    
    # Use a custom service config with the free port
    service_config = ServiceConfig(
        host="localhost",
        port=port,
        max_workers=10,
        models_dir="trained_models",
        checkpoint_dir="checkpoints",
        log_level="INFO"
    )
    
    model_store = ModelStore(
        models_dir=service_config.models_dir,
        checkpoint_dir=service_config.checkpoint_dir
    )
    
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=10000
    )
    
    # Create server
    server = create_server(
        orchestrator=orchestrator,
        model_store=model_store,
        config=service_config,
        adapter_factory=None
    )
    
    # Start server in background thread
    def run_server():
        server.start()
        server.wait_for_termination()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(2)
    
    try:
        # Create client and check health
        address = f"{service_config.host}:{service_config.port}"
        with grpc.insecure_channel(address) as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            
            health_stub = health_pb2_grpc.HealthStub(channel)
            request = health_pb2.HealthCheckRequest()
            response = health_stub.Check(request, timeout=10)  # type: ignore[attr-defined]
            
            # Verify server is healthy
            assert response.status == health_pb2.HealthCheckResponse.SERVING
    
    finally:
        # Cleanup
        server.stop(grace_period=1)
