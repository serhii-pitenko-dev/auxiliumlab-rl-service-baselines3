"""Training orchestration, callbacks, and run management."""
import logging
import threading
from typing import Dict, Optional
import uuid
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np

from .dto import TrainingConfig, RunInfo, RunStatus
from .algorithms import build_model, get_model_class
from .env import ExternalSimEnv
from ..infra.model_store import ModelStore
from ..infra.external_env_adapter import ExternalEnvAdapter, FakeExternalEnvAdapter, GrpcExternalEnvAdapter
from ..infra.config import EnvConfig

logger = logging.getLogger(__name__)


class CheckpointCallback(BaseCallback):
    """Callback for saving periodic checkpoints during training."""
    
    def __init__(
        self,
        model_store: ModelStore,
        run_id: str,
        checkpoint_freq: int = 10000,
        run_registry: Optional[Dict[str, RunInfo]] = None,
        verbose: int = 0
    ):
        """
        Initialize the checkpoint callback.
        
        Args:
            model_store: Model store for saving checkpoints
            run_id: Unique run identifier
            checkpoint_freq: Save checkpoint every N timesteps
            run_registry: Registry to update progress
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.model_store = model_store
        self.run_id = run_id
        self.checkpoint_freq = checkpoint_freq
        self.run_registry = run_registry
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.n_calls % self.checkpoint_freq == 0:
            checkpoint_path = self.model_store.save_checkpoint(
                self.model,
                self.run_id,
                self.num_timesteps
            )
            logger.info(f"[{self.run_id}] Checkpoint saved at timestep {self.num_timesteps}")
            
            # Update run registry
            if self.run_registry and self.run_id in self.run_registry:
                with threading.Lock():
                    self.run_registry[self.run_id].last_checkpoint_path = checkpoint_path
                    self.run_registry[self.run_id].timesteps_done = self.num_timesteps
        
        # Update progress in registry
        if self.run_registry and self.run_id in self.run_registry:
            with threading.Lock():
                self.run_registry[self.run_id].timesteps_done = self.num_timesteps
        
        return True  # Continue training


class TrainingOrchestrator:
    """Manages training runs and provides thread-safe access to run status."""
    
    def __init__(
        self,
        model_store: ModelStore,
        env_config: EnvConfig,
        checkpoint_freq: int = 10000
    ):
        """
        Initialize the training orchestrator.
        
        Args:
            model_store: Model store for saving models and checkpoints
            env_config: Environment configuration
            checkpoint_freq: Checkpoint frequency in timesteps
        """
        self.model_store = model_store
        self.env_config = env_config
        self.checkpoint_freq = checkpoint_freq
        
        # Thread-safe run registry
        self.run_registry: Dict[str, RunInfo] = {}
        self.registry_lock = threading.Lock()
    
    def start_training(
        self,
        config: TrainingConfig,
        adapter: Optional[ExternalEnvAdapter] = None
    ) -> str:
        """
        Start a training run asynchronously.
        
        Args:
            config: Training configuration
            adapter: Optional external environment adapter (uses Fake if not provided)
            
        Returns:
            Unique run ID
        """
        run_id = str(uuid.uuid4())
        
        # Create run info
        run_info = RunInfo(
            run_id=run_id,
            config=config,
            status=RunStatus.PENDING,
            total_timesteps=config.total_timesteps
        )
        
        # Register the run
        with self.registry_lock:
            self.run_registry[run_id] = run_info
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._train_worker,
            args=(run_id, config, adapter),
            daemon=True
        )
        thread.start()
        
        logger.info(f"Started training run {run_id} for {config.algorithm.value.upper()}")
        return run_id
    
    def _train_worker(
        self,
        run_id: str,
        config: TrainingConfig,
        adapter: Optional[ExternalEnvAdapter]
    ) -> None:
        """Background worker for training."""
        try:
            # Update status to running
            with self.registry_lock:
                self.run_registry[run_id].status = RunStatus.RUNNING
            
            # Build the environment.
            # If .NET passed gym_ids in hyperparameters, create one adapter per gym;
            # for multiple gyms wrap them in a DummyVecEnv so SB3 can exploit
            # parallelism.  If no gym_ids are provided fall back to the fake adapter.
            obs_dim = self.env_config.observation_dim
            act_dim = self.env_config.action_dim
            max_steps = self.env_config.max_steps

            gym_ids_str = config.hyperparameters.get("gym_ids", "")
            gym_ids = [g for g in gym_ids_str.split(";") if g]

            if len(gym_ids) > 1:
                # Multiple gyms — use the vectorised wrapper
                def _make_env(gid: str):
                    def _init():
                        a = GrpcExternalEnvAdapter("localhost:50062", gym_id=gid)
                        return ExternalSimEnv(
                            adapter=a,
                            observation_dim=obs_dim,
                            action_dim=act_dim,
                            max_steps=max_steps
                        )
                    return _init
                env = DummyVecEnv([_make_env(gid) for gid in gym_ids])
                logger.info(f"[{run_id}] Created DummyVecEnv with {len(gym_ids)} gym(s)")
            else:
                # Single gym — use the adapter provided by the servicer (already has
                # the correct gym_id) or a fresh real/fake adapter.
                if adapter is None:
                    if gym_ids:
                        adapter = GrpcExternalEnvAdapter("localhost:50062", gym_id=gym_ids[0])
                        logger.info(f"[{run_id}] Created single adapter for gym_id={gym_ids[0]}")
                    else:
                        adapter = FakeExternalEnvAdapter(
                            observation_dim=obs_dim,
                            action_dim=act_dim
                        )
                        logger.info(f"[{run_id}] No gym_ids found — using FakeExternalEnvAdapter")
                env = ExternalSimEnv(
                    adapter=adapter,
                    observation_dim=obs_dim,
                    action_dim=act_dim,
                    max_steps=max_steps
                )
            
            # Build model
            hyperparams = config.get_hyperparams_typed()
            model = build_model(
                algorithm=config.algorithm,
                env=env,
                hyperparameters=hyperparams,
                seed=config.seed,
                verbose=1
            )
            
            # Create callbacks
            checkpoint_callback = CheckpointCallback(
                model_store=self.model_store,
                run_id=run_id,
                checkpoint_freq=self.checkpoint_freq,
                run_registry=self.run_registry
            )
            
            # Train
            logger.info(f"[{run_id}] Starting training for {config.total_timesteps} timesteps")
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=checkpoint_callback,
                progress_bar=False
            )
            
            # Save final model
            model_path = config.model_output_path or f"./trained_models/{config.experiment_id}_{run_id}.zip"
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            final_path = self.model_store.save_model(model, run_id, config.experiment_id)
            
            # Update status to completed
            with self.registry_lock:
                self.run_registry[run_id].status = RunStatus.COMPLETED
                self.run_registry[run_id].final_model_path = final_path
                self.run_registry[run_id].timesteps_done = config.total_timesteps
                self.run_registry[run_id].model_in_memory = model  # Keep for inference
            
            logger.info(f"[{run_id}] Training completed successfully")
            
            # Clean up
            env.close()
            
        except Exception as e:
            logger.error(f"[{run_id}] Training failed: {e}", exc_info=True)
            with self.registry_lock:
                self.run_registry[run_id].status = RunStatus.FAILED
                self.run_registry[run_id].error_message = str(e)
    
    def get_run_info(self, run_id: str) -> Optional[RunInfo]:
        """
        Get information about a training run.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Run information, or None if not found
        """
        with self.registry_lock:
            return self.run_registry.get(run_id)
    
    def predict(self, run_id: str, observation: np.ndarray) -> Optional[int]:
        """
        Perform inference with a trained model.
        
        Args:
            run_id: Unique run identifier
            observation: Observation to predict action for
            
        Returns:
            Predicted action, or None if model not available
        """
        with self.registry_lock:
            run_info = self.run_registry.get(run_id)
            if not run_info:
                return None
            
            # Try in-memory model first
            if run_info.model_in_memory:
                action, _ = run_info.model_in_memory.predict(observation, deterministic=True)
                return int(action)
            
            # Try loading from disk
            if run_info.final_model_path:
                model_class = get_model_class(run_info.config.algorithm)
                model = self.model_store.load_model(run_info.final_model_path, model_class)
                action, _ = model.predict(observation, deterministic=True)
                return int(action)
        
        return None
