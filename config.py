"""
Muon vs AdamW Fine-Tuning Experiment Configuration
Based on Moonlight scaling law research (arXiv:2502.16982)

Key parameters from paper:
- Muon lr: 0.02 for hidden weights, 3e-4 for scalars/biases
- AdamW lr: 3e-4 default
- Weight decay: 0.01 for both
- Batch size: 2M tokens (large batches benefit Muon most)
"""

from dataclasses import dataclass
from enum import Enum

class OptimizerType(Enum):
    MUON = "adamw_torch"
    ADAMW = "adamw_torch"
    HYBRID = "adamw_torch"  # Muon for hidden, AdamW for scalars/biases

@dataclass
class ExperimentConfig:
    """Configuration for Muon vs AdamW comparison"""
    
    # Model Configuration
    model_name: str = "Qwen/Qwen3.5-35B-A3B"
    max_seq_length: int = 4096
    
    # Dataset (for demonstration - use small subset)
    dataset_name: str = "alpaca_cleaned"
    train_samples: int = 1000  # Reduced for testing
    test_samples: int = 200
    
    # Training Hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 1  # Limited by VRAM
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    
    # Muon-specific parameters (from Moonlight paper)
    muon_lr_hidden: float = 0.02
    muon_lr_scalars: float = 3e-4
    weight_decay: float = 0.01
    num_ns_steps: int = 5  # Newton-Schulz iterations
    
    # AdamW parameters (standard)
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.95
    
    # Evaluation metrics
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Output paths
    output_dir: str = "experiments/moon_vs_adamw"
    checkpoint_dir: str = f"{output_dir}/checkpoints"
    log_dir: str = f"{output_dir}/logs"
    results_file: str = f"{output_dir}/results.csv"

@dataclass
class MuonConfig(ExperimentConfig):
    """Configuration for Muon optimizer"""
    
    # Parameter grouping strategy (from Moonlight paper)
    split_strategy: str = "by_ndim_and_name"  # Hidden weights vs scalars/biases
    
    # Newton-Schulz orthogonalization settings
    ns_coefficients: dict = None  # Will be set to quintic coefficients
    
    def __post_init__(self):
        if self.ns_coefficients is None:
            self.ns_coefficients = {
                'a': 3.4445,
                'b': -4.7750,
                'c': 2.0315
            }

@dataclass 
class AdamWConfig(ExperimentConfig):
    """Configuration for standard AdamW optimizer"""
    
    # Standard AdamW parameters
    max_grad_norm: float = 1.0

config_moonlight = MuonConfig()
config_adamw = AdamWConfig()

if __name__ == "__main__":
    print("=== Moon vs AdamW Experiment Configuration ===")
    print(f"Model: {config_moonlight.model_name}")
    print(f"Moonlight lr (hidden): {config_moonlight.muon_lr_hidden}")
    print(f"Moonlight lr (scalars): {config_moonlight.muon_lr_scalars}")
    print(f"Weight decay: {config_moonlight.weight_decay}")
    print(f"Newton-Schulz steps: {config_moonlight.num_ns_steps}")
