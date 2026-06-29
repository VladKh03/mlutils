from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm.auto import tqdm

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
	"""
	Returns the current learning rate from the optimizer
	"""

	return optimizer.param_groups[0]["lr"]


def calculate_topk_correct(
	logits: torch.Tensor,
	labels: torch.Tensor,
	k: int = 5
) -> int:
	"""
	Calculates how many labels are inside top-k predictions
	"""

	if k <= 1:
		preds = logits.argmax(dim=1)
		return int((preds == labels).sum().item())

	k = min(k, logits.shape[1])

	topk_preds = torch.topk(
		logits,
		k=k,
		dim=1
	).indices

	correct = topk_preds.eq(labels.view(-1, 1)).any(dim=1).sum().item()

	return int(correct)


def train_step(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	loss_fn: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	scaler: Optional[torch.amp.GradScaler] = None,
	grad_clip: Optional[float] = None,
	use_amp: bool = False
) -> Tuple[float, float]:
	"""
	Trains a model for one epoch

	Returns:
		train_loss: Average training loss
		train_acc: Average training top-1 accuracy
	"""

	model.train()

	total_loss = 0.0
	total_correct = 0
	total_samples = 0

	for X, y in dataloader:
		X = X.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		batch_size = X.shape[0]

		optimizer.zero_grad(set_to_none=True)

		if use_amp and scaler is not None:
			with torch.amp.autocast(
				device_type=device.type,
				enabled=device.type == "cuda"
			):
				logits = model(X)
				loss = loss_fn(logits, y)

			scaler.scale(loss).backward()

			if grad_clip is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					max_norm=grad_clip
				)

			scaler.step(optimizer)
			scaler.update()

		else:
			logits = model(X)
			loss = loss_fn(logits, y)

			loss.backward()

			if grad_clip is not None:
				torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					max_norm=grad_clip
				)

			optimizer.step()

		total_loss += loss.item() * batch_size
		total_correct += (logits.argmax(dim=1) == y).sum().item()
		total_samples += batch_size

	train_loss = total_loss / total_samples
	train_acc = total_correct / total_samples

	return train_loss, train_acc


def test_step(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	loss_fn: torch.nn.Module,
	device: torch.device,
	top_k: int = 5,
	use_amp: bool = False
) -> Tuple[float, float, float]:
	"""
	Tests a model for one epoch

	Returns:
		test_loss: Average test loss
		test_acc: Average test top-1 accuracy
		test_topk_acc: Average test top-k accuracy
	"""

	model.eval()

	total_loss = 0.0
	total_top1_correct = 0
	total_topk_correct = 0
	total_samples = 0

	with torch.inference_mode():
		for X, y in dataloader:
			X = X.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)

			batch_size = X.shape[0]

			with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp and device.type == "cuda"
            ):
				logits = model(X)
				loss = loss_fn(logits, y)

			total_loss += loss.item() * batch_size
			total_top1_correct += (logits.argmax(dim=1) == y).sum().item()
			total_topk_correct += calculate_topk_correct(
				logits=logits,
				labels=y,
				k=top_k
			)
			total_samples += batch_size

	test_loss = total_loss / total_samples
	test_acc = total_top1_correct / total_samples
	test_topk_acc = total_topk_correct / total_samples

	return test_loss, test_acc, test_topk_acc


def save_checkpoint(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	epoch: int,
	results: Dict[str, List[float]],
	save_path: str | Path,
	test_loss: float,
	test_acc: float
) -> None:
	"""
	Saves model checkpoint
	"""

	save_path = Path(save_path)
	save_path.parent.mkdir(parents=True, exist_ok=True)

	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"results": results,
		"test_loss": test_loss,
		"test_acc": test_acc
	}

	torch.save(checkpoint, save_path)


def train(
	model: torch.nn.Module,
	train_dataloader: torch.utils.data.DataLoader,
	test_dataloader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	loss_fn: torch.nn.Module,
	epochs: int,
	device: torch.device,
	scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
	top_k: int = 5,
	use_amp: bool = True,
	grad_clip: Optional[float] = None,
	save_best_path: Optional[str | Path] = None,
	save_last_path: Optional[str | Path] = None,
	save_best_by: str = "test_loss"
) -> Dict[str, List[float]]:
	"""
	Trains and tests a PyTorch model

	Args:
		model: PyTorch model
		train_dataloader: DataLoader for training data
		test_dataloader: DataLoader for test data
		optimizer: PyTorch optimizer
		loss_fn: Loss function
		epochs: Number of epochs
		device: Device, for example "cuda" or "cpu"
		scheduler: Optional learning rate scheduler
		top_k: Top-k accuracy value
		use_amp: Use mixed precision on CUDA
		grad_clip: Max gradient norm. If None, gradient clipping is disabled
		save_best_path: Path for saving the best model checkpoint
		save_last_path: Path for saving the last model checkpoint
		save_best_by: "test_loss" or "test_acc"

	Returns:
		Dictionary with training history
	"""

	if save_best_by not in {"test_loss", "test_acc"}:
		raise ValueError("save_best_by must be 'test_loss' or 'test_acc'")

	model.to(device)

	amp_enabled = use_amp and device.type == "cuda"

	scaler = torch.amp.GradScaler(
		device="cuda",
		enabled=amp_enabled
	)

	results = {
		"train_loss": [],
		"train_acc": [],
		"test_loss": [],
		"test_acc": [],
		f"test_top{top_k}_acc": [],
		"lr": []
	}

	best_test_loss = float("inf")
	best_test_acc = 0.0

	for epoch in tqdm(range(epochs)):
		train_loss, train_acc = train_step(
			model=model,
			dataloader=train_dataloader,
			loss_fn=loss_fn,
			optimizer=optimizer,
			device=device,
			scaler=scaler,
			grad_clip=grad_clip,
			use_amp=amp_enabled
		)

		test_loss, test_acc, test_topk_acc = test_step(
			model=model,
			dataloader=test_dataloader,
			loss_fn=loss_fn,
			device=device,
			top_k=top_k,
			use_amp=amp_enabled
		)

		if scheduler is not None:
			if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				scheduler.step(test_loss)
			else:
				scheduler.step()

		current_lr = get_current_lr(optimizer)

		results["train_loss"].append(train_loss)
		results["train_acc"].append(train_acc)
		results["test_loss"].append(test_loss)
		results["test_acc"].append(test_acc)
		results[f"test_top{top_k}_acc"].append(test_topk_acc)
		results["lr"].append(current_lr)

		should_save_best = False

		if save_best_by == "test_loss" and test_loss < best_test_loss:
			best_test_loss = test_loss
			should_save_best = True

		if save_best_by == "test_acc" and test_acc > best_test_acc:
			best_test_acc = test_acc
			should_save_best = True

		if should_save_best and save_best_path is not None:
			save_checkpoint(
				model=model,
				optimizer=optimizer,
				epoch=epoch + 1,
				results=results,
				save_path=save_best_path,
				test_loss=test_loss,
				test_acc=test_acc
			)

		print(
			f"Epoch: {epoch + 1} | "
			f"lr: {current_lr:.8f} | "
			f"train_loss: {train_loss:.4f} | "
			f"train_acc: {train_acc:.4f} | "
			f"test_loss: {test_loss:.4f} | "
			f"test_acc: {test_acc:.4f} | "
			f"test_top{top_k}_acc: {test_topk_acc:.4f}"
		)

	if save_last_path is not None:
		save_checkpoint(
			model=model,
			optimizer=optimizer,
			epoch=epochs,
			results=results,
			save_path=save_last_path,
			test_loss=results["test_loss"][-1],
			test_acc=results["test_acc"][-1]
		)

	return results