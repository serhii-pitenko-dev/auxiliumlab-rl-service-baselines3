"""Model storage and checkpoint management."""
import os
from pathlib import Path
from typing import Optional
import logging
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


class ModelStore:
    """Handles saving and loading of trained models and checkpoints."""
    
    def __init__(self, models_dir: str = "./trained_models", checkpoint_dir: str = "./checkpoints"):
        """
        Initialize the model store.
        
        Args:
            models_dir: Directory for final trained models
            checkpoint_dir: Directory for intermediate checkpoints
        """
        self.models_dir = Path(models_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: BaseAlgorithm, run_id: str, experiment_id: str) -> str:
        """
        Save a trained model.
        
        Args:
            model: The trained SB3 model
            run_id: Unique run identifier
            experiment_id: Experiment identifier
            
        Returns:
            Path to the saved model
        """
        model_path = self.models_dir / f"{experiment_id}_{run_id}.zip"
        model.save(str(model_path))
        logger.info(f"Saved model to {model_path}")
        return str(model_path)
    
    def save_checkpoint(self, model: BaseAlgorithm, run_id: str, timestep: int) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to checkpoint
            run_id: Unique run identifier
            timestep: Current training timestep
            
        Returns:
            Path to the checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}_checkpoint_{timestep}.zip"
        model.save(str(checkpoint_path))
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_model(self, model_path: str, model_class: type) -> BaseAlgorithm:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
            model_class: The SB3 model class (PPO, A2C, DQN)
            
        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = model_class.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    
    def get_latest_checkpoint(self, run_id: str) -> Optional[str]:
        """
        Get the path to the latest checkpoint for a run.
        
        Args:
            run_id: Unique run identifier
            
        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob(f"{run_id}_checkpoint_*.zip"))
        if not checkpoints:
            return None
        
        # Sort by timestep number (extracted from filename)
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return str(latest)
