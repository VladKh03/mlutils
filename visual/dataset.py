from pathlib import Path
from typing import Optional

import random
import matplotlib.pyplot as plt
import torch

from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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

	Args:
		image_tensor: Image tensor with shape [C, H, W]
		mean: Normalization mean
		std: Normalization std

	Returns:
		Denormalized image tensor
	"""

	if mean is None or std is None:
		return image_tensor

	mean_tensor = torch.tensor(mean).view(-1, 1, 1)
	std_tensor = torch.tensor(std).view(-1, 1, 1)

	image_tensor = image_tensor.cpu() * std_tensor + mean_tensor
	image_tensor = image_tensor.clamp(0, 1)

	return image_tensor


def plot_tensor_image(
	image_tensor: torch.Tensor,
	title: Optional[str] = None,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Plots one image tensor.

	Args:
		image_tensor: Image tensor with shape [C, H, W]
		title: Optional plot title
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
	"""

	if image_tensor.ndim != 3:
		raise ValueError("image_tensor must have shape [C, H, W]")

	image_tensor = _denormalize_image(
		image_tensor=image_tensor,
		mean=mean,
		std=std
	)

	image = image_tensor.permute(1, 2, 0).cpu().numpy()

	plt.figure(figsize=(5, 5))
	plt.imshow(image)
	plt.axis("off")

	if title is not None:
		plt.title(title)

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)


def plot_random_images(
	dataset,
	class_names: Optional[list[str]] = None,
	num_images: int = 9,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	figsize: tuple[int, int] = (10, 10),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150,
	seed: Optional[int] = None
) -> None:
	"""
	Plots random images from a PyTorch dataset

	Args:
		dataset: PyTorch dataset
		class_names: Optional list of class names
		num_images: Number of images to show
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
		seed: Optional random seed
	"""

	if num_images <= 0:
		raise ValueError("num_images must be greater than 0")

	if len(dataset) == 0:
		raise ValueError("dataset is empty")

	if seed is not None:
		random.seed(seed)

	num_images = min(num_images, len(dataset))
	indices = random.sample(range(len(dataset)), num_images)

	cols = int(num_images ** 0.5)

	if cols * cols < num_images:
		cols += 1

	rows = (num_images + cols - 1) // cols

	plt.figure(figsize=figsize)

	for plot_index, dataset_index in enumerate(indices):
		image, label = dataset[dataset_index]

		if not isinstance(image, torch.Tensor):
			raise TypeError("dataset must return image as torch.Tensor")

		image = _denormalize_image(
			image_tensor=image,
			mean=mean,
			std=std
		)

		image = image.permute(1, 2, 0).cpu().numpy()

		plt.subplot(rows, cols, plot_index + 1)
		plt.imshow(image)
		plt.axis("off")

		if class_names is not None:
			plt.title(class_names[int(label)])
		else:
			plt.title(str(label))

	plt.tight_layout()

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)

def plot_image_transform(
	image_path: str | Path,
	transform: transforms.Compose,
	mean: Optional[list[float]] = None,
	std: Optional[list[float]] = None,
	figsize: tuple[int, int] = (10, 5),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Plots original image and transformed image

	Args:
		image_path: Path to image
		transform: Transform to apply
		mean: Optional mean for denormalization
		std: Optional std for denormalization
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
	"""

	image_path = Path(image_path)

	if not image_path.exists():
		raise FileNotFoundError(f"Image does not exist: {image_path}")

	image = Image.open(image_path).convert("RGB")
	transformed_image = transform(image)

	if not isinstance(transformed_image, torch.Tensor):
		raise TypeError(
			"transform must return torch.Tensor"
			"Make sure your transform includes transforms.ToTensor()"
		)

	transformed_image = _denormalize_image(
		image_tensor=transformed_image,
		mean=mean,
		std=std
	)

	transformed_image = transformed_image.permute(1, 2, 0).cpu().numpy()

	plt.figure(figsize=figsize)

	plt.subplot(1, 2, 1)
	plt.imshow(image)
	plt.title("Original")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(transformed_image)
	plt.title("Transformed")
	plt.axis("off")

	plt.tight_layout()

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)


def count_images_by_class(
	data_dir: str | Path,
	image_extensions: set[str] = IMAGE_EXTENSIONS
) -> dict[str, int]:
	"""
	Counts images in each class folder

	Args:
		data_dir: Dataset folder with class folders
		image_extensions: Image extensions to count

	Returns:
		Dictionary with class names and image counts
	"""

	data_dir = Path(data_dir)

	if not data_dir.exists():
		raise FileNotFoundError(f"Directory does not exist: {data_dir}")

	if not data_dir.is_dir():
		raise NotADirectoryError(f"Path is not a directory: {data_dir}")

	image_extensions = {
		extension.lower()
		for extension in image_extensions
	}

	class_counts = {}

	for class_dir in data_dir.iterdir():
		if not class_dir.is_dir():
			continue

		count = 0

		for file_path in class_dir.rglob("*"):
			if file_path.is_file() and file_path.suffix.lower() in image_extensions:
				count += 1

		class_counts[class_dir.name] = count

	return class_counts


def plot_class_distribution(
	data_dir: str | Path,
	top_n: Optional[int] = None,
	sort: bool = True,
	figsize: tuple[int, int] = (12, 8),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Plots number of images in each class

	Args:
		data_dir: Dataset folder with class folders
		top_n: Optional number of classes to show
		sort: If True, sorts classes by image count
		figsize: Figure size
		save_path: Optional path to save the plot
		show: If True, shows the plot
		dpi: Image quality when saving
	"""

	class_counts = count_images_by_class(data_dir)

	if len(class_counts) == 0:
		raise ValueError(f"No class folders found in: {data_dir}")

	items = list(class_counts.items())

	if sort:
		items = sorted(
			items,
			key=lambda item: item[1],
			reverse=True
		)

	if top_n is not None:
		if top_n <= 0:
			raise ValueError("top_n must be greater than 0")

		items = items[:top_n]

	class_names = [item[0] for item in items]
	counts = [item[1] for item in items]

	plt.figure(figsize=figsize)
	plt.barh(class_names, counts)
	plt.xlabel("Image count")
	plt.ylabel("Class")
	plt.title("Class distribution")
	plt.gca().invert_yaxis()
	plt.grid(True, axis="x", alpha=0.3)

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)