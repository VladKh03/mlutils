from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class TrainerConfig:
  """
  Configuration for the Trainer class

  Args:
    epochs (int): Number of epochs to train the model
    device (torch.device): Device used for training, CUDA if available, otherwise CPU
    use_amp (bool): Use automatic mixed precision during training
    compile_model (bool): Compile the model before training
    compile_mode (str | None): Mode used by torch.compile. If None, PyTorch uses the default mode
    grad_clip (Optional[float]): Maximum gradient norm for gradient clipping. If None, clipping is not used
    use_tqdm (bool): Show a tqdm progress bar during training
    validate_every (int): Run validation every N epochs
    epoch_scheduler_monitor (str): Metric name used by an epoch scheduler(ReduceLROnPlateau), for example "test_loss" | "test_acc" | "train_loss" | "train_acc" | "top_NUM_acc"
  """

  epochs: int
  device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
  )

  use_amp: bool = False
  compile_model: bool = False
  compile_mode: str | None = None

  grad_clip: Optional[float] = None
  use_tqdm: bool = True
  validate_every: int  = 1
  epoch_scheduler_monitor: str = "test_loss"
