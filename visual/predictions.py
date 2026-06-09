from pathlib import Path
from typing import Optional, Any

import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms
import random

def _save_or_show_plot(
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Saves and/or shows current matplotlib plot
	"""

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

	if show:
		plt.show()
	else:
		plt.close()


def _denormalize_image(
	image_tensor: torch.Tensor,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None
) -> torch.Tensor:
	"""
	Denormalizes image tensor for visualization
	"""

	if mean is None or std is None:
		return image_tensor.clamp(0, 1)

	mean_tensor = torch.tensor(mean).view(-1, 1, 1)
	std_tensor = torch.tensor(std).view(-1, 1, 1)

	image_tensor = image_tensor.cpu() * std_tensor + mean_tensor
	image_tensor = image_tensor.clamp(0, 1)

	return image_tensor


def _tensor_to_image(
	image_tensor: torch.Tensor,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None
):
	"""
	Converts image tensor [C, H, W] to NumPy image [H, W, C]
	"""

	if image_tensor.ndim != 3:
		raise ValueError("image_tensor must have shape [C, H, W]")

	image_tensor = _denormalize_image(
		image_tensor=image_tensor,
		mean=mean,
		std=std
	)

	return image_tensor.permute(1, 2, 0).cpu().numpy()


def _predict_tensor_topk(
	model: torch.nn.Module,
	image_tensor: torch.Tensor,
	class_names: list[str],
	device: torch.device,
	top_k: int = 5
) -> list[dict[str, Any]]:
	"""
	Gets top-k predictions for one image tensor
	"""

	if top_k <= 0:
		raise ValueError("top_k must be greater than 0")

	model.eval()
	model.to(device)

	if image_tensor.ndim == 3:
		image_tensor = image_tensor.unsqueeze(dim=0)

	if image_tensor.ndim != 4:
		raise ValueError("image_tensor must have shape [3, H, W] or [1, 3, H, W]")

	image_tensor = image_tensor.to(device, non_blocking=True)

	with torch.inference_mode():
		logits = model(image_tensor)
		probs = torch.softmax(logits, dim=1)

	top_k = min(top_k, probs.shape[1])

	top_probs, top_indices = torch.topk(
		probs,
		k=top_k,
		dim=1
	)

	predictions = []

	for prob, index in zip(top_probs[0], top_indices[0]):
		class_index = int(index.item())

		predictions.append({
			"class_index": class_index,
			"class_name": class_names[class_index],
			"probability": float(prob.item())
		})

	return predictions


def plot_prediction_bar(
	predictions: list[dict[str, Any]],
	title: str = "Top predictions",
	figsize: tuple[int, int] = (8, 5),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Plots top-k predictions as a horizontal bar chart

	Args:
		predictions: List returned by predict function
		title: Plot title
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
	"""

	if len(predictions) == 0:
		raise ValueError("predictions cannot be empty")

	class_names = [
		prediction["class_name"]
		for prediction in predictions
	]

	probabilities = [
		prediction["probability"] * 100
		for prediction in predictions
	]

	plt.figure(figsize=figsize)
	plt.barh(class_names, probabilities)
	plt.xlabel("Probability (%)")
	plt.title(title)
	plt.gca().invert_yaxis()
	plt.grid(True, axis="x", alpha=0.3)

	for index, probability in enumerate(probabilities):
		plt.text(
			probability,
			index,
			f" {probability:.2f}%",
			va="center"
		)

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)


def plot_image_prediction(
	model: torch.nn.Module,
	image_path: str | Path,
	transform: transforms.Compose,
	class_names: list[str],
	device: torch.device,
	top_k: int = 5,
	true_class_name: Optional[str] = None,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	figsize: tuple[int, int] = (12, 5),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> list[dict[str, Any]]:
	"""
	Plots one image with top-k model predictions

	Args:
		model: PyTorch model
		image_path: Path to image
		transform: Transform for the image
		class_names: List of class names
		device: Device, for example torch.device("cuda")
		top_k: Number of top predictions
		true_class_name: Optional true class name
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving

	Returns:
		List with top-k predictions.
	"""

	image_path = Path(image_path)

	if not image_path.exists():
		raise FileNotFoundError(f"Image does not exist: {image_path}")

	image = Image.open(image_path).convert("RGB")
	transformed_image = transform(image)

	if not isinstance(transformed_image, torch.Tensor):
		raise TypeError(
			"transform must return torch.Tensor. "
			"Make sure your transform includes transforms.ToTensor()."
		)

	predictions = _predict_tensor_topk(
		model=model,
		image_tensor=transformed_image,
		class_names=class_names,
		device=device,
		top_k=top_k
	)

	image_for_plot = _tensor_to_image(
		image_tensor=transformed_image,
		mean=mean,
		std=std
	)

	predicted_class = predictions[0]["class_name"]
	predicted_probability = predictions[0]["probability"] * 100

	plt.figure(figsize=figsize)

	plt.subplot(1, 2, 1)
	plt.imshow(image_for_plot)
	plt.axis("off")

	if true_class_name is not None:
		title = (
			f"True: {true_class_name}\n"
			f"Pred: {predicted_class} ({predicted_probability:.2f}%)"
		)
	else:
		title = f"Pred: {predicted_class} ({predicted_probability:.2f}%)"

	plt.title(title)

	plt.subplot(1, 2, 2)

	class_labels = [
		prediction["class_name"]
		for prediction in predictions
	]

	probabilities = [
		prediction["probability"] * 100
		for prediction in predictions
	]

	plt.barh(class_labels, probabilities)
	plt.xlabel("Probability (%)")
	plt.title(f"Top-{len(predictions)} predictions")
	plt.gca().invert_yaxis()
	plt.grid(True, axis="x", alpha=0.3)

	for index, probability in enumerate(probabilities):
		plt.text(
			probability,
			index,
			f" {probability:.2f}%",
			va="center"
		)

	plt.tight_layout()

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)

	return predictions


def plot_predictions_grid(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	class_names: list[str],
	device: torch.device,
	num_images: int = 16,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	figsize: tuple[int, int] = (12, 12),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150,
	random_images: bool = True,
	seed: Optional[int] = None
) -> None:
	"""
	Plots images with true and predicted labels

	Args:
		model: PyTorch model
		dataloader: DataLoader with images and labels
		class_names: List of class names
		device: Device, for example torch.device("cuda")
		num_images: Number of images to show
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
		random_images: If True, selects random images from dataset
		seed: Optional random seed
	"""

	if num_images <= 0:
		raise ValueError("num_images must be greater than 0")

	model.eval()
	model.to(device)

	if random_images:
		dataset = dataloader.dataset

		if len(dataset) == 0:
			raise ValueError("dataloader dataset is empty")

		if seed is not None:
			random.seed(seed)

		num_images = min(num_images, len(dataset))
		indices = random.sample(range(len(dataset)), num_images)

		images_list = []
		labels_list = []

		for index in indices:
			image, label = dataset[index]

			if not isinstance(image, torch.Tensor):
				raise TypeError("dataset must return image as torch.Tensor")

			images_list.append(image)
			labels_list.append(label)

		images = torch.stack(images_list)
		labels = torch.as_tensor(labels_list)

	else:
		images, labels = next(iter(dataloader))
		num_images = min(num_images, images.shape[0])
		images = images[:num_images]
		labels = labels[:num_images]

	X = images.to(device, non_blocking=True)

	with torch.inference_mode():
		logits = model(X)
		probs = torch.softmax(logits, dim=1)
		preds = probs.argmax(dim=1)
		confidences = probs.max(dim=1).values

	cols = int(num_images ** 0.5)

	if cols * cols < num_images:
		cols += 1

	rows = (num_images + cols - 1) // cols

	plt.figure(figsize=figsize)

	for index in range(num_images):
		image = _tensor_to_image(
			image_tensor=images[index].cpu(),
			mean=mean,
			std=std
		)

		true_index = int(labels[index].item())
		pred_index = int(preds[index].cpu().item())
		confidence = float(confidences[index].cpu().item()) * 100

		true_class = class_names[true_index]
		pred_class = class_names[pred_index]

		plt.subplot(rows, cols, index + 1)
		plt.imshow(image)
		plt.axis("off")

		title = (
			f"True: {true_class}\n"
			f"Pred: {pred_class}\n"
			f"{confidence:.1f}%"
		)

		plt.title(title, fontsize=9)

	plt.tight_layout()

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)


def plot_wrong_predictions(
	model: torch.nn.Module,
	dataloader: torch.utils.data.DataLoader,
	class_names: list[str],
	device: torch.device,
	num_images: int = 16,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	figsize: tuple[int, int] = (12, 12),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150,
	most_confident: bool = True
) -> None:
	"""
	Plots wrong predictions from a DataLoader

	Args:
		model: PyTorch model
		dataloader: DataLoader with images and labels
		class_names: List of class names
		device: Device, for example torch.device("cuda")
		num_images: Number of wrong predictions to show
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
		most_confident: If True, shows most confident wrong predictions
	"""

	if num_images <= 0:
		raise ValueError("num_images must be greater than 0")

	model.eval()
	model.to(device)

	wrong_predictions = []

	with torch.inference_mode():
		for X, y in dataloader:
			X = X.to(device, non_blocking=True)
			y = y.to(device, non_blocking=True)

			logits = model(X)
			probs = torch.softmax(logits, dim=1)
			preds = probs.argmax(dim=1)
			confidences = probs.max(dim=1).values

			wrong_mask = preds != y
			wrong_indices = torch.where(wrong_mask)[0]

			for index in wrong_indices:
				wrong_predictions.append({
					"image": X[index].cpu(),
					"true_index": int(y[index].cpu().item()),
					"pred_index": int(preds[index].cpu().item()),
					"confidence": float(confidences[index].cpu().item())
				})

	if len(wrong_predictions) == 0:
		print("[INFO] No wrong predictions found.")
		return

	if most_confident:
		wrong_predictions = sorted(
			wrong_predictions,
			key=lambda item: item["confidence"],
			reverse=True
		)

	wrong_predictions = wrong_predictions[:num_images]

	cols = int(len(wrong_predictions) ** 0.5)

	if cols * cols < len(wrong_predictions):
		cols += 1

	rows = (len(wrong_predictions) + cols - 1) // cols

	plt.figure(figsize=figsize)

	for index, item in enumerate(wrong_predictions):
		image = _tensor_to_image(
			image_tensor=item["image"],
			mean=mean,
			std=std
		)

		true_class = class_names[item["true_index"]]
		pred_class = class_names[item["pred_index"]]
		confidence = item["confidence"] * 100

		plt.subplot(rows, cols, index + 1)
		plt.imshow(image)
		plt.axis("off")

		plt.title(
			f"True: {true_class}\n"
			f"Pred: {pred_class}\n"
			f"{confidence:.1f}%",
			fontsize=9
		)

	plt.tight_layout()

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)