from .config import TrainerConfig
from .metrics import Metric, LossMetric
from .callbacks import Callback
from typing import Optional, List, Dict
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

def reset_metrics(metrics: List[Metric]):
  """
  Reset all metrics before a new epoch
  """

  for metric in metrics:
    metric.reset()

def update_metrics(
  metrics: List[Metric],
  logits: torch.Tensor,
  targets: torch.Tensor,
  loss: torch.Tensor
):
  """
  Update all metrics using one batch
  """

  for metric in metrics:
    metric.update(
      logits=logits.detach(),
      targets=targets,
      loss=loss.detach()
    )

def compute_metrics(metrics: List[Metric]) -> Dict[str, float]:
  """
  Compute all metrics and return one result dictionary
  """

  results = {}

  for metric in metrics:
    results.update(metric.compute())

  return results


class Trainer:
  def __init__(
    self,
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
    test_dataloader: Optional[DataLoader] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    train_metrics: Optional[List[Metric]] = None,
    val_metrics: Optional[List[Metric]] = None,
    callbacks: Optional[List[Callback]] = None
  ):
    """
    Create a Trainer for a PyTorch model

    The Trainer handles the full training loop
    It can run training, validation, metrics, callbacks, AMP, gradient clipping,
    model compilation, and scheduler steps

    Args:
      model (torch.nn.Module): Model to train
      train_dataloader (DataLoader): Dataloader used for training
      loss_fn (torch.nn.Module): Loss function used to compute loss
      optimizer (torch.optim.Optimizer): Optimizer used to update model weights
      config (TrainerConfig): Training configuration
      test_dataloader (Optional[DataLoader]): Dataloader used for validation
      scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]): Learning rate scheduler
      train_metrics (Optional[List[Metric]]): Metrics used during training
      val_metrics (Optional[List[Metric]]): Metrics used during validation
      callbacks (Optional[List[Callback]]): Callbacks called during training
    """

    self.model = model
    self.train_dataloader = train_dataloader
    self.test_dataloader = test_dataloader
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.config = config

    self.scaler = None
    self.is_prepared = False
    self.scheduler = scheduler

    self.train_metrics = train_metrics or [
      LossMetric("train_loss")
    ]
    self.val_metrics = val_metrics or (
      [LossMetric("test_loss")] if test_dataloader is not None else []
    )
    self.callbacks = callbacks or []

    self.current_epoch = 0
    self.current_epoch_results = {}
    self.results = {}
    self.should_stop = False


  def compile_model(self):
    """
    Compile the model with torch.compile
    """

    return torch.compile(
      model=self.model,
      mode=self.config.compile_mode
    )

  def set_scaler(self):
    """
    Create AMP gradient scaler

    The scaler is enabled only when AMP is used on CUDA
    """

    return torch.amp.GradScaler( # pyright: ignore[reportPrivateImportUsage]
      device="cuda",
      enabled=self.config.use_amp and self.config.device.type == "cuda"
    )

  def do_clip_grad_norm(self):
    """
    Clip model gradients if gradient clipping is enabled
    """
      
    torch.nn.utils.clip_grad_norm_(
      self.model.parameters(), # pyright: ignore[reportFunctionMemberAccess]
      self.config.grad_clip # pyright: ignore[reportArgumentType]
    )

  def get_logits_loss(self, X, y):
    """
    Run forward pass and compute loss
    """

    logits = self.model(X)

    return logits, self.loss_fn(logits, y)

  def get_logits_loss_amp_autocast(self, X, y):
    """
    Run forward pass and compute loss with AMP autocast
    """

    with torch.amp.autocast( # pyright: ignore[reportPrivateImportUsage]
      device_type=self.config.device.type,
      enabled=self.config.use_amp and self.config.device.type == "cuda"
    ):
      return self.get_logits_loss(X, y)

  def append_results(self, input_results):
    """
    Add current epoch results to the full training history
    """

    for key, value in input_results.items():
      if key not in self.results:
        self.results[key] = []

      self.results[key].append(value)

  # Callback manager
  def call_callbacks(self, hook_name: str):
    """
    Call one hook on all callbacks
    """

    for callback in self.callbacks:
      hook = getattr(callback, hook_name)
      hook(self)

  def prepare(self):
    """
    Prepare the trainer before training starts

    Moves the model to device, checks config values, compiles the model if needed,
    creates AMP scaler, and marks the trainer as prepared
    """

    self.model.to(self.config.device) # pyright: ignore[reportFunctionMemberAccess]

    if self.config.validate_every < 1:
      raise ValueError("validate_every must be >= 1")

    if self.config.compile_model:
      self.model = self.compile_model()

    self.scaler = self.set_scaler()

    self.is_prepared = True

  def move_batch(self, batch):
    """
    Move one batch to the configured device

    Args:
      batch: Batch in the format X, y

    Returns:
      tuple: X and y moved to device
    """

    X, y = batch

    return (
      X.to(self.config.device, non_blocking=True),
      y.to(self.config.device, non_blocking=True)
    )

  def step_scheduler(self):
    """
    Step the learning rate scheduler after an epoch

    ReduceLROnPlateau uses the metric from current epoch results
    Other schedulers are stepped without a metric
    """

    if self.scheduler is None:
      return

    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
      monitor = self.config.epoch_scheduler_monitor

      if monitor not in self.current_epoch_results:
        raise ValueError(
          f"ReduceLROnPlateau requires monitor='{monitor}', "
          f"but available metrics are {list(self.current_epoch_results.keys())}"
        )

      self.scheduler.step(self.current_epoch_results[monitor])
    else:
      self.scheduler.step()

  def train_one_epoch(self):
    """
    Run one training epoch

    Updates model weights and returns training metrics
    """

    self.model.train() # pyright: ignore[reportFunctionMemberAccess]

    reset_metrics(self.train_metrics)

    for batch in self.train_dataloader:
      X, y = self.move_batch(batch)

      self.optimizer.zero_grad(
        set_to_none=True
      )

      if self.config.use_amp and self.config.device.type == "cuda":
        logits, loss = self.get_logits_loss_amp_autocast(X, y)

        self.scaler.scale(loss).backward() # pyright: ignore[reportOptionalMemberAccess]

        if self.config.grad_clip is not None:
          self.scaler.unscale_(self.optimizer) # pyright: ignore[reportOptionalMemberAccess]
          self.do_clip_grad_norm()

        self.scaler.step(self.optimizer) # pyright: ignore[reportOptionalMemberAccess]
        self.scaler.update() # pyright: ignore[reportOptionalMemberAccess]

      else:
        logits, loss = self.get_logits_loss(X, y)

        loss.backward()

        if self.config.grad_clip is not None:
          self.do_clip_grad_norm()

        self.optimizer.step()

      update_metrics(
        metrics=self.train_metrics,
        logits=logits,
        targets=y,
        loss=loss
      )

    return compute_metrics(self.train_metrics)

  def validate_one_epoch(self):
    """
    Run one validation epoch

    Does not update model weights
    Returns validation metrics
    """

    self.model.eval() # pyright: ignore[reportFunctionMemberAccess]
    reset_metrics(self.val_metrics)

    with torch.inference_mode():
      for batch in self.test_dataloader: # pyright: ignore[reportOptionalIterable]
        X, y = self.move_batch(batch)

        if self.config.use_amp and self.config.device.type == "cuda":
          logits, loss = self.get_logits_loss_amp_autocast(X, y)
        else:
          logits, loss = self.get_logits_loss(X, y)

        update_metrics(
          metrics=self.val_metrics,
          logits=logits,
          targets=y,
          loss=loss
        )

    return compute_metrics(self.val_metrics)

  def fit(self):
    """
    Run the full training loop

    Calls callbacks, trains the model, optionally runs validation,
    steps the scheduler, saves results, and stops early if needed

    Returns:
      Dict[str, List[float]]: Training history with metric names and values
    """
    
    if not self.is_prepared:
      self.prepare()

    self.results = {}

    epoch_iterator = range(self.config.epochs)

    if self.config.use_tqdm:
      epoch_iterator = tqdm(
        epoch_iterator
      )

    self.call_callbacks("on_train_start")

    for epoch in epoch_iterator:
      self.current_epoch = epoch
      self.current_epoch_results = {}

      self.call_callbacks("on_epoch_start")

      train_results = self.train_one_epoch()

      self.current_epoch_results.update(train_results)

      self.append_results(train_results)

      if (
        self.test_dataloader is not None and
        (epoch + 1) % self.config.validate_every == 0
      ):
        self.call_callbacks("on_validation_start")

        val_results = self.validate_one_epoch()

        self.current_epoch_results.update(val_results)

        self.append_results(val_results)

        self.call_callbacks("on_validation_end")

      self.call_callbacks("on_epoch_end")

      self.step_scheduler()

      if self.should_stop:
        break

    self.call_callbacks("on_train_end")

    return self.results
