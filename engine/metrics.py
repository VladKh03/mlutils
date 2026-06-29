from abc import ABC, abstractmethod
from typing import Dict

import torch

class Metric(ABC):
  """
  Base class for all training metrics

  Each metric stores its own state during one epoch
  The Trainer calls reset, update, and compute during training or validation

  Args:
    name (str): Metric name used in the results dictionary
  """

  def __init__(self, name: str):
    self.name = name

  @abstractmethod
  def reset(self):
    pass

  @abstractmethod
  def update(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor
  ):
    pass

  @abstractmethod
  def compute(self) -> Dict[str, float]:
    pass


class LossMetric(Metric):
  """
  Metric that computes average loss over one epoch

  It stores the sum of batch losses and divides it by the total number of samples

  Args:
    name (str): Metric name used in the results dictionary
  """

  def __init__(self, name: str = "loss"):
    super().__init__(name)
    self.reset()

  def reset(self):
    """
    Reset loss state before a new epoch
    """

    self.loss_sum = None
    self.total = 0

  def update(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor
  ):
    """
    Add loss value from one batch

    Args:
      logits (torch.Tensor): Model outputs before softmax
      targets (torch.Tensor): True class labels
      loss (torch.Tensor): Loss value for the current batch
    """

    batch_size = targets.size(0)

    # Keep loss on device to avoid GPU sync from .item()
    batch_loss_sum = loss.detach() * batch_size

    if self.loss_sum is None:
      self.loss_sum = batch_loss_sum
    else:
      self.loss_sum += batch_loss_sum

    self.total += batch_size

  def compute(self) -> Dict[str, float]:
    """
    Compute average loss for the epoch

    Returns:
      Dict[str, float]: Dictionary with metric name and average loss
    """

    if self.total == 0:
      return {self.name: 0.0}

    return {
      self.name: (self.loss_sum / self.total).item() # pyright: ignore[reportOptionalOperand]
    }


class AccuracyMetric(Metric):
  """
  Metric that computes classification accuracy over one epoch

  It counts correct predictions and divides them by the total number of samples

  Args:
    name (str): Metric name used in the results dictionary
  """

  def __init__(self, name: str = "acc"):
    super().__init__(name)
    self.reset()

  def reset(self):
    """
    Reset accuracy state before a new epoch
    """

    self.correct = None
    self.total = 0

  def update(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor
  ):
    """
    Add accuracy values from one batch

    Args:
      logits (torch.Tensor): Model outputs before softmax
      targets (torch.Tensor): True class labels
      loss (torch.Tensor): Loss value for the current batch
    """

    preds = logits.argmax(dim=1)
    batch_correct = (preds == targets).sum().detach()

    if self.correct is None:
      self.correct = batch_correct
    else:
      self.correct += batch_correct

    self.total += targets.size(0)

  def compute(self) -> Dict[str, float]:
    """
    Compute accuracy for the epoch

    Returns:
      Dict[str, float]: Dictionary with metric name and accuracy value
    """

    if self.total == 0:
      return {self.name: 0.0}

    return {
      self.name: (self.correct.float() / self.total).item() # pyright: ignore[reportOptionalMemberAccess]
    }


class TopKAccuracyMetric(Metric):
  """
  Metric that computes top-k classification accuracy over one epoch

  Prediction is counted as correct if the true label is inside top-k predictions

  Args:
    k (int): Number of top predictions to check
  """

  def __init__(self, k: int = 5):
    self.k = k
    super().__init__(f"top_{self.k}_acc")
    self.reset()

  def reset(self):
    """
    Reset top-k accuracy state before a new epoch
    """

    self.correct = None
    self.total = 0

  def update(
    self,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor
  ):
    """
    Add top-k accuracy values from one batch

    Args:
      logits (torch.Tensor): Model outputs before softmax
      targets (torch.Tensor): True class labels
      loss (torch.Tensor): Loss value for the current batch
    """

    k = min(self.k, logits.shape[1])

    topk_preds = torch.topk(
      logits,
      k=k,
      dim=1
    ).indices

    correct = topk_preds.eq(targets.view(-1, 1)).any(dim=1)

    # Keep value on device to avoid GPU sync from .item()
    batch_correct = correct.sum().detach()

    if self.correct is None:
      self.correct = batch_correct
    else:
      self.correct += batch_correct

    self.total += targets.size(0)

  def compute(self) -> Dict[str, float]:
    """
    Compute top-k accuracy for the epoch

    Returns:
      Dict[str, float]: Dictionary with metric name and top-k accuracy value
    """

    if self.total == 0:
      return {self.name: 0.0}

    return {
      self.name: (self.correct.float() / self.total).item() # pyright: ignore[reportOptionalMemberAccess]
    }
