from pathlib import Path
import torch

class Callback():
  """
  Base class for callbacks 
  """

  def on_train_start(self, trainer):
    """
    Do something before training starts
    """
    pass

  def on_train_end(self, trainer):
    """
    Do something after training ends
    """
    pass

  def on_validation_start(self, trainer):
    """
    Do something before validation(testing) starts
    """
    pass

  def on_validation_end(self, trainer):
    """
    Do something after validation(testing) ends
    """
    pass

  def on_epoch_start(self, trainer):
    """
    Do something before epoch starts
    """
    pass

  def on_epoch_end(self, trainer):
    """
    Do something after epoch ends
    """
    pass

class PrintMetricsCallback(Callback):
  """
  Callback that prints metrics at the end of each epoch

  It prints the current epoch number and all metrics from Trainer
  """
  def on_epoch_end(self, trainer):
    epoch = trainer.current_epoch + 1
    metrics = trainer.current_epoch_results

    parts = [
      f"Epoch: {epoch}"
    ]

    for key, value in metrics.items():
      if isinstance(value, float):
        parts.append(f"{key}: {value:.4f}")
      else:
        parts.append(f"{key}: {value}")
    print(" | ".join(parts))

class EarlyStoppingCallback(Callback):
  """
  Callback that stops training when a metric stops improving

  It checks the monitored metric at the end of each epoch
  If the metric does not improve for several epochs, training is stopped

  Args:
    monitor (str): Metric name to watch, for example "test_loss" or "test_acc". Default to "test_loss"
    patience (int): Number of epochs to wait before stopping. Default to 5
    min_delta (float): Minimum change needed to count as improvement. Default to 0.0
    mode (str): Use "min" when lower is better, or "max" when higher is better. Default to "min"
    verbose (bool): Print a message when training is stopped. Default to True
  """

  def __init__(
    self,
    monitor: str = "test_loss",
    patience: int = 5,
    min_delta: float = 0.0,
    mode: str = "min",
    verbose: bool = True
  ):
    if mode not in ["min", "max"]:
      raise ValueError("Mode must be min or max")

    self.monitor = monitor
    self.patience = patience
    self.min_delta = min_delta
    self.mode = mode
    self.verbose = verbose

    self.best_value: float | None = None
    self.wait = 0
    self.best_epoch = 0

  def on_train_start(self, trainer):
    """
    Reset callback state before training starts
    """

    self.best_value = None
    self.wait = 0
    self.best_epoch = 0

  def on_epoch_end(self, trainer):
    """
    Check the monitored metric at the end of each epoch

    Stops training if the metric did not improve for too many epochs
    """

    metrics = trainer.current_epoch_results

    if self.monitor not in metrics:
      return
    
    current_value = metrics[self.monitor]

    if isinstance(current_value, torch.Tensor):
      current_value = current_value.item()

    current_value = float(current_value)

    if self.best_value is None:
      self.best_value = current_value
      self.best_epoch = trainer.current_epoch + 1
      return

    if self._is_improvement(current_value):
      self.best_value = current_value
      self.wait = 0
      self.best_epoch = trainer.current_epoch + 1
    else:
      self.wait += 1

    if self.wait >= self.patience:
      trainer.should_stop = True

      if self.verbose:
        print(
          f"Early stopping at epoch {trainer.current_epoch + 1} | "
          f"Best epoch: {self.best_epoch} | "
          f"best {self.monitor}: {self.best_value:.4f}"
        )

  def _is_improvement(self, current_value: float) -> bool:
    """
    Return True if the current value is better that the best value
    """

    if self.best_value is None:
      return True

    if self.mode == "min":
      return current_value < self.best_value - self.min_delta

    return current_value > self.best_value + self.min_delta

class SaveBestModelCallback(Callback):
  """
  Callback that saves the best model checkpoint during training

  It checks one metric at the end of each epoch
  If the metric improves, it saves the model, optimizer, results, and training state

  Args:
    save_path (str | Path): Path where the checkpoint will be saved
    skip_epochs (int): Number of first epochs to skip before saving. Default to 0
    monitor (str): Metric name to watch, for example "test_loss" or "test_acc". Default to "test_loss"
    mode (str): Use "min" when lower is better, or "max" when higher is better. Default to "min"
  """

  def __init__(
    self,
    save_path: str | Path,
    skip_epochs: int = 0,
    monitor: str = "test_loss",
    mode: str = "min"
  ):
    if mode not in ["min", "max"]:
      raise ValueError("mode must be 'min' or 'max'")

    self.save_path = Path(save_path)
    self.skip_epochs = skip_epochs
    self.monitor = monitor
    self.mode = mode

    self.best_value: float | None = None

  def on_train_start(self, trainer):
    """
    Reset callback state before training starts
    """
    self.best_value = None

  def on_epoch_end(self, trainer):
    """
    Save a checkpoint if the monitored metric improved
    """
    metrics = trainer.current_epoch_results

    if self.monitor not in metrics:
      return

    current_value = metrics[self.monitor]

    if isinstance(current_value, torch.Tensor):
      current_value = current_value.item()

    current_value = float(current_value)

    if trainer.current_epoch + 1 <= self.skip_epochs:
      return

    if self.best_value is None or self._is_improvement(current_value):
      self.best_value = current_value
      self.save_checkpoint(trainer, current_value)

  def _is_improvement(self, current_value: float) -> bool:
    """
    Return True if the current value is better than the best value
    """
    if self.best_value is None:
      return True
    
    if self.mode == "min":
      return current_value < self.best_value

    return current_value > self.best_value

  def save_checkpoint(self, trainer, current_value: float) -> None:
    """
    Save model, optimizer, results, and training state to disk
    """
    self.save_path.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = getattr(trainer.model, "_orig_mod", trainer.model)

    checkpoint = {
      "epoch": trainer.current_epoch + 1,
      "model_state_dict": model_to_save.state_dict(),
      "optimizer_state_dict": trainer.optimizer.state_dict(),
      "results": trainer.results,
      "metrics": trainer.current_epoch_results,
      "monitor": self.monitor,
      "best_value": current_value,
      "mode": self.mode,
    }

    if getattr(trainer, "scaler", None) is not None:
      checkpoint["scaler_state_dict"] = trainer.scaler.state_dict()

    if getattr(trainer, "scheduler", None) is not None:
      checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

    return torch.save(checkpoint, self.save_path)