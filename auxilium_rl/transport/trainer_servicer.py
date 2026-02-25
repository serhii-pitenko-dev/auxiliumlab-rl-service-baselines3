"""gRPC servicer implementation for the Policy Trainer service."""
import logging
from typing import Optional, Callable
import numpy as np

# Import generated gRPC code
from generated import policy_trainer_pb2, policy_trainer_pb2_grpc

from ..core.dto import TrainingConfig, AlgorithmType, RunStatus
from ..core.training import TrainingOrchestrator
from ..infra.external_env_adapter import ExternalEnvAdapter

logger = logging.getLogger(__name__)


class PolicyTrainerServicer(policy_trainer_pb2_grpc.PolicyTrainerServiceServicer):
    """Implementation of the PolicyTrainerService."""
    
    def __init__(
        self,
        orchestrator: TrainingOrchestrator,
        adapter_factory: Optional[Callable[[], ExternalEnvAdapter]] = None
    ):
        """
        Initialize the servicer.
        
        Args:
            orchestrator: Training orchestrator
            adapter_factory: Optional factory function to create external adapters
        """
        self.orchestrator = orchestrator
        self.adapter_factory = adapter_factory
    
    def StartTrainingPPO(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context
    ) -> policy_trainer_pb2.TrainingResponse:
        """Start training a PPO model."""
        return self._start_training(request, AlgorithmType.PPO)
    
    def StartTrainingA2C(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context
    ) -> policy_trainer_pb2.TrainingResponse:
        """Start training an A2C model."""
        return self._start_training(request, AlgorithmType.A2C)
    
    def StartTrainingDQN(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        context
    ) -> policy_trainer_pb2.TrainingResponse:
        """Start training a DQN model."""
        return self._start_training(request, AlgorithmType.DQN)
    
    def _start_training(
        self,
        request: policy_trainer_pb2.TrainingRequest,
        algorithm: AlgorithmType
    ) -> policy_trainer_pb2.TrainingResponse:
        """Common training start logic."""
        try:
            # Validate request
            if not request.experiment_id:
                return policy_trainer_pb2.TrainingResponse(
                    status=policy_trainer_pb2.FAILED,
                    message="experiment_id is required",
                    run_id=""
                )
            
            if request.total_timesteps <= 0:
                return policy_trainer_pb2.TrainingResponse(
                    status=policy_trainer_pb2.FAILED,
                    message="total_timesteps must be positive",
                    run_id=""
                )
            
            # Create training config
            config = TrainingConfig(
                algorithm=algorithm,
                experiment_id=request.experiment_id,
                total_timesteps=request.total_timesteps,
                seed=request.seed,
                hyperparameters=dict(request.hyperparameters),
                model_output_path=request.model_output_path
            )
            
            # Create adapter if factory is provided
            adapter = None
            if self.adapter_factory:
                adapter = self.adapter_factory()
            
            # Start training
            run_id = self.orchestrator.start_training(config, adapter)
            
            logger.info(
                f"Training started: run_id={run_id}, algorithm={algorithm.value}, "
                f"experiment={request.experiment_id}, timesteps={request.total_timesteps}"
            )
            
            return policy_trainer_pb2.TrainingResponse(
                status=policy_trainer_pb2.STARTED,
                message=f"Training started successfully for {algorithm.value.upper()}",
                run_id=run_id
            )
            
        except Exception as e:
            logger.error(f"Failed to start training: {e}", exc_info=True)
            return policy_trainer_pb2.TrainingResponse(
                status=policy_trainer_pb2.FAILED,
                message=f"Failed to start training: {str(e)}",
                run_id=""
            )
    
    def GetTrainingStatus(
        self,
        request: policy_trainer_pb2.StatusRequest,
        context
    ) -> policy_trainer_pb2.StatusResponse:
        """Get the status of a training run."""
        run_info = self.orchestrator.get_run_info(request.run_id)
        
        if not run_info:
            return policy_trainer_pb2.StatusResponse(
                timesteps_done=0,
                is_done=False,
                last_checkpoint_path="",
                error_message=f"Run {request.run_id} not found"
            )
        
        is_done = run_info.status in [RunStatus.COMPLETED, RunStatus.FAILED]
        
        return policy_trainer_pb2.StatusResponse(
            timesteps_done=run_info.timesteps_done,
            is_done=is_done,
            last_checkpoint_path=run_info.last_checkpoint_path or "",
            error_message=run_info.error_message or ""
        )
    
    def Act(
        self,
        request: policy_trainer_pb2.ActRequest,
        context
    ) -> policy_trainer_pb2.ActResponse:
        """Perform inference with a trained model."""
        try:
            # Convert observation to numpy array
            observation = np.array(request.observation, dtype=np.float32)
            
            # Get prediction
            action = self.orchestrator.predict(request.run_id, observation)
            
            if action is None:
                return policy_trainer_pb2.ActResponse(
                    action=0,
                    success=False,
                    error_message=f"Model not available for run {request.run_id}"
                )
            
            return policy_trainer_pb2.ActResponse(
                action=action,
                success=True,
                error_message=""
            )
            
        except Exception as e:
            logger.error(f"Failed to perform inference: {e}", exc_info=True)
            return policy_trainer_pb2.ActResponse(
                action=0,
                success=False,
                error_message=str(e)
            )
