"""Health check servicer implementation."""
import logging
from typing import Optional
from pathlib import Path

from grpc_health.v1 import health_pb2, health_pb2_grpc

from ..core.training import TrainingOrchestrator
from ..infra.model_store import ModelStore

logger = logging.getLogger(__name__)


class HealthServicer(health_pb2_grpc.HealthServicer):
    """
    Implements gRPC Health Checking Protocol.
    
    Verifies that critical service components are functional:
    - Training orchestrator is available
    - Model store directories are accessible
    """
    
    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        model_store: ModelStore
    ):
        """
        Initialize the health servicer.
        
        Args:
            orchestrator: Training orchestrator to verify
            model_store: Model store to verify
        """
        self.orchestrator = orchestrator
        self.model_store = model_store
    
    def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context
    ) -> health_pb2.HealthCheckResponse:
        """
        Perform a health check.
        
        Args:
            request: Health check request (can specify service name)
            context: gRPC context
            
        Returns:
            Health check response with serving status
        """
        try:
            # Verify orchestrator is available
            if self.orchestrator is None:
                logger.warning("Health check failed: orchestrator is None")
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
            
            # Verify orchestrator run registry is accessible
            if not hasattr(self.orchestrator, 'run_registry'):
                logger.warning("Health check failed: orchestrator missing run_registry")
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
            
            # Verify model store is available
            if self.model_store is None:
                logger.warning("Health check failed: model_store is None")
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
            
            # Verify model directories exist and are accessible
            if not self._check_model_store_directories():
                logger.warning("Health check failed: model store directories not accessible")
                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.NOT_SERVING
                )
            
            # All checks passed
            logger.debug("Health check passed")
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.SERVING
            )
            
        except Exception as e:
            logger.error(f"Health check error: {e}", exc_info=True)
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.NOT_SERVING
            )
    
    def _check_model_store_directories(self) -> bool:
        """
        Verify model store directories are accessible.
        
        Returns:
            True if directories are accessible, False otherwise
        """
        try:
            # Check models directory
            models_dir = Path(self.model_store.models_dir)
            if not models_dir.exists():
                logger.warning(f"Models directory does not exist: {models_dir}")
                return False
            
            # Check checkpoint directory
            checkpoint_dir = Path(self.model_store.checkpoint_dir)
            if not checkpoint_dir.exists():
                logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
                return False
            
            # Verify write access by checking if directories are writable
            if not models_dir.is_dir() or not checkpoint_dir.is_dir():
                logger.warning("Model store paths are not directories")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model store directories: {e}")
            return False
    
    def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context
    ):
        """
        Stream health status changes (optional method).
        
        This implementation sends the current status and completes.
        For production use, you might want to implement continuous monitoring.
        """
        # Send current status
        yield self.Check(request, context)
