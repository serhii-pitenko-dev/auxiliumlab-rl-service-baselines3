"""
Server entry point for the RL Training Service.

This script starts the gRPC server and handles graceful shutdown.
"""
import sys
import os

# The gRPC-generated stubs use flat imports (e.g. `import policy_trainer_pb2`)
# rather than package-qualified ones. Add the generated/ directory to sys.path
# so Python can resolve those imports regardless of the working directory.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated"))

import signal
import logging

from auxilium_rl.infra.config import ServiceConfig, EnvConfig
from auxilium_rl.infra.external_env_adapter import GrpcExternalEnvAdapter
from auxilium_rl.infra.logging import setup_logging
from auxilium_rl.infra.model_store import ModelStore
from auxilium_rl.core.training import TrainingOrchestrator
from auxilium_rl.transport.grpc_server import create_server

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    # Load configuration
    service_config = ServiceConfig.from_env()
    env_config = EnvConfig.from_env()
    
    # Setup logging
    setup_logging(level=service_config.log_level)
    logger.info("Starting RL Training Service...")
    
    # Create model store
    model_store = ModelStore(
        models_dir=service_config.models_dir,
        checkpoint_dir=service_config.checkpoint_dir
    )
    
    # Create training orchestrator
    orchestrator = TrainingOrchestrator(
        model_store=model_store,
        env_config=env_config,
        checkpoint_freq=10000
    )
    
    # Create and start gRPC server
    server = create_server(
        orchestrator=orchestrator,
        model_store=model_store,
        config=service_config,
        adapter_factory=lambda gym_id: GrpcExternalEnvAdapter("localhost:50062", gym_id=gym_id)  # Connect to C# SimulationService
    )
    
    # Setup signal handlers for graceful shutdown
    def shutdown_handler(signum, frame):
        logger.info("Received shutdown signal")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    # Start server
    try:
        server.start()
        logger.info("Server is ready to accept requests")
        server.wait_for_termination()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
