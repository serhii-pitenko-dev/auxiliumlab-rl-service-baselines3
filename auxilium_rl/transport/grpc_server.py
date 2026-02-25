"""gRPC server setup and lifecycle management."""
import logging
from concurrent import futures
from typing import Callable, Optional
import grpc
from grpc_health.v1 import health_pb2_grpc

from generated import policy_trainer_pb2_grpc
from .trainer_servicer import PolicyTrainerServicer
from .health_servicer import HealthServicer
from ..core.training import TrainingOrchestrator
from ..infra.config import ServiceConfig
from ..infra.model_store import ModelStore

logger = logging.getLogger(__name__)


class GrpcServer:
    """gRPC server wrapper."""
    
    def __init__(
        self,
        servicer: PolicyTrainerServicer,
        health_servicer: HealthServicer,
        config: ServiceConfig
    ):
        """
        Initialize the gRPC server.
        
        Args:
            servicer: Policy trainer servicer
            health_servicer: Health check servicer
            config: Service configuration
        """
        self.servicer = servicer
        self.health_servicer = health_servicer
        self.config = config
        self.server = None
    
    def start(self) -> None:
        """Start the gRPC server."""
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.config.max_workers)
        )
        
        # Add servicer to server
        policy_trainer_pb2_grpc.add_PolicyTrainerServiceServicer_to_server(
            self.servicer,
            self.server
        )
        
        # Add health check servicer
        health_pb2_grpc.add_HealthServicer_to_server(
            self.health_servicer,
            self.server
        )
        
        # Bind to address
        address = f"{self.config.host}:{self.config.port}"
        self.server.add_insecure_port(address)
        
        # Start server
        self.server.start()
        logger.info(f"gRPC server started on {address} with health check enabled")
    
    def wait_for_termination(self) -> None:
        """Block until server termination."""
        if self.server:
            self.server.wait_for_termination()
    
    def stop(self, grace_period: int = 5) -> None:
        """
        Stop the gRPC server.
        
        Args:
            grace_period: Grace period in seconds for shutdown
        """
        if self.server:
            logger.info("Stopping gRPC server...")
            self.server.stop(grace_period)
            logger.info("gRPC server stopped")


def create_server(
    orchestrator: TrainingOrchestrator,
    model_store: ModelStore,
    config: ServiceConfig,
    adapter_factory: Optional[Callable] = None
) -> GrpcServer:
    """
    Factory function to create a configured gRPC server.
    
    Args:
        orchestrator: Training orchestrator
        model_store: Model store for health checks
        config: Service configuration
        adapter_factory: Optional factory for creating external adapters
        
    Returns:
        Configured gRPC server
    """
    servicer = PolicyTrainerServicer(orchestrator, adapter_factory)
    health_servicer = HealthServicer(orchestrator, model_store)
    return GrpcServer(servicer, health_servicer, config)
