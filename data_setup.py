import os
import zipfile
import requests
import random
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Iterable
from urllib.parse import urlparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def _safe_extract_zip(zip_ref: zipfile.ZipFile, extract_to: Path) -> None:
	extract_to = extract_to.resolve()

	for member in zip_ref.namelist():
		member_path = (extract_to / member).resolve()

		if not str(member_path).startswith(str(extract_to)):
			raise ValueError(f"Unsafe zip file path detected: {member}")

	zip_ref.extractall(extract_to)


def download_data(
	source: str,
	destination: Union[str, Path],
	base_dir: Union[str, Path] = "data",
	remove_source: bool = True,
	force_download: bool = False,
	timeout: int = 30,
	chunk_size: int = 8192
) -> Path:
	"""
	Downloads a zip dataset and extracts it to a folder

	Args:
		source: URL to the zip file
		destination: Folder name or path for extracted data
		base_dir: Main folder where data will be saved
		remove_source: Remove the zip file after extraction
		force_download: Download again even if destination exists
		timeout: Request timeout in seconds
		chunk_size: Download chunk size in bytes

	Returns:
		Path to the extracted dataset folder
	"""

	base_dir = Path(base_dir)
	destination = Path(destination)

	data_path = base_dir
	output_path = data_path / destination

	data_path.mkdir(parents=True, exist_ok=True)

	if output_path.exists() and not force_download:
		print(f"[INFO] {output_path} already exists. Skipping download.")
		return output_path

	output_path.mkdir(parents=True, exist_ok=True)

	parsed_url = urlparse(source)
	target_file_name = Path(parsed_url.path).name

	if not target_file_name:
		raise ValueError("Could not detect file name from source URL.")

	if not target_file_name.endswith(".zip"):
		raise ValueError(f"Source file must be a .zip file. Got: {target_file_name}")

	target_file_path = data_path / target_file_name

	print(f"[INFO] Downloading {target_file_name} from {source}...")

	try:
		response = requests.get(
			source,
			stream=True,
			timeout=timeout
		)
		response.raise_for_status()

		with open(target_file_path, "wb") as file:
			for chunk in response.iter_content(chunk_size=chunk_size):
				if chunk:
					file.write(chunk)

		print(f"[INFO] Extracting {target_file_name} to {output_path}...")

		with zipfile.ZipFile(target_file_path, "r") as zip_ref:
			_safe_extract_zip(zip_ref, output_path)

		if remove_source:
			target_file_path.unlink(missing_ok=True)
			print(f"[INFO] Removed zip file: {target_file_path}")

	except Exception as error:
		if target_file_path.exists():
			target_file_path.unlink(missing_ok=True)

		raise RuntimeError(f"Failed to download or extract data: {error}") from error

	return output_path

def delete_corrupted_images(
	data_dir: str | Path,
	image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
	dry_run: bool = False
) -> int:
	"""
	Checks image files in a folder and deletes corrupted images

	Args:
		data_dir: Path to the image dataset folder
		image_extensions: Image file extensions to check
		dry_run: If True, only prints corrupted images without deleting them

	Returns:
		Number of corrupted images found
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

	corrupted_count = 0

	for image_path in data_dir.rglob("*"):
		if not image_path.is_file():
			continue

		if image_path.suffix.lower() not in image_extensions:
			continue

		try:
			with Image.open(image_path) as image:
				image.verify()

			with Image.open(image_path) as image:
				image.load()

		except (UnidentifiedImageError, OSError, ValueError) as error:
			corrupted_count += 1

			if dry_run:
				print(f"[BAD] {image_path} | {error}")
			else:
				print(f"[DELETE] {image_path} | {error}")
				image_path.unlink()

	print(f"[INFO] Checked folder: {data_dir}")
	print(f"[INFO] Corrupted images found: {corrupted_count}")

	if dry_run:
		print("[INFO] dry_run=True, no files were deleted.")

	return corrupted_count

def create_train_test_split(
	source_dir: str | Path,
	output_dir: str | Path,
	train_percent: float = 0.8,
	image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
	seed: int = 42,
	move_files: bool = False,
	overwrite: bool = False
) -> Path:
	"""
	Creates train and test folders from one image dataset folder
	The source folder must have class folders inside it:
		source_dir/
			class_1/
			...

	The output folder will have this structure:
		output_dir/
			train/
				class_1/
			test/
				class_1/

	Args:
		source_dir: Path to the original dataset folder
		output_dir: Path where train and test folders will be created
		train_percent: Part of images for train. Example: 0.8 means 80% train and 20% test
		image_extensions: Image file extensions to include
		seed: Random seed for reproducible split
		move_files: If True, moves files. If False, copies files
		overwrite: If True, overwrites existing files

	Returns:
		Path to the output dataset folder
	"""

	source_dir = Path(source_dir)
	output_dir = Path(output_dir)

	if not source_dir.exists():
		raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

	if not source_dir.is_dir():
		raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

	if not 0 < train_percent < 1:
		raise ValueError("train_percent must be greater than 0 and less than 1")

	image_extensions = {
		extension.lower()
		for extension in image_extensions
	}

	class_dirs = [
		path
		for path in source_dir.iterdir()
		if path.is_dir()
	]

	if len(class_dirs) == 0:
		raise ValueError(f"No class folders found in: {source_dir}")

	random.seed(seed)

	train_dir = output_dir / "train"
	test_dir = output_dir / "test"

	train_dir.mkdir(parents=True, exist_ok=True)
	test_dir.mkdir(parents=True, exist_ok=True)

	for source_class_dir in class_dirs:
		class_name = source_class_dir.name

		train_class_dir = train_dir / class_name
		test_class_dir = test_dir / class_name

		train_class_dir.mkdir(parents=True, exist_ok=True)
		test_class_dir.mkdir(parents=True, exist_ok=True)

		images = [
			file
			for file in source_class_dir.iterdir()
			if file.is_file() and file.suffix.lower() in image_extensions
		]

		if len(images) == 0:
			print(f"[WARNING] {class_name}: no images found")
			continue

		random.shuffle(images)

		train_count = int(len(images) * train_percent)

		train_images = images[:train_count]
		test_images = images[train_count:]

		for image_path in train_images:
			target_path = train_class_dir / image_path.name

			if target_path.exists() and not overwrite:
				continue

			if move_files:
				shutil.move(str(image_path), str(target_path))
			else:
				shutil.copy2(image_path, target_path)

		for image_path in test_images:
			target_path = test_class_dir / image_path.name

			if target_path.exists() and not overwrite:
				continue

			if move_files:
				shutil.move(str(image_path), str(target_path))
			else:
				shutil.copy2(image_path, target_path)

		action = "moved" if move_files else "copied"

		print(
			f"[INFO] {class_name}: "
			f"{action} {len(train_images)} train images, "
			f"{action} {len(test_images)} test images"
		)

	return output_dir

def create_folder_dataloaders(
	train_dir: str | Path,
	test_dir: str | Path,
	train_transform: transforms.Compose,
	test_transform: transforms.Compose,
	batch_size: int,
	num_workers: Optional[int] = None,
	pin_memory: Optional[bool] = None,
	persistent_workers: Optional[bool] = None,
	drop_last: bool = False,
	shuffle_train: bool = True,
	shuffle_test: bool = False,
	seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, List[str], Dict[str, int]]:
	"""
	Creates DataLoaders for train and test image folders
	The function uses ImageFolder, so the folders must have this structure:
		train_dir/
			class_1/
			...
		test_dir/
			class_1/
			...

	Args:
		train_dir: Path to the training folder
		test_dir: Path to the test folder
		train_transform: Transformations for training images
		test_transform: Transformations for test images
		batch_size: Number of images in one batch
		num_workers: Number of workers for loading data
		pin_memory: Use pinned memory if CUDA is available
		persistent_workers: Keep workers alive between epochs
		drop_last: Drop the last incomplete training batch
		shuffle_train: Shuffle training data
		shuffle_test: Shuffle test data
		seed: Seed for reproducible shuffling

	Returns:
		train_dataloader: DataLoader for training data
		test_dataloader: DataLoader for test data
		class_names: List of class names
		class_to_idx: Dictionary with class names and class indexes
	"""
	train_dir = Path(train_dir)
	test_dir = Path(test_dir)

	if not train_dir.exists():
		raise FileNotFoundError(f"Train directory does not exist: {train_dir}")

	if not test_dir.exists():
		raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

	if batch_size <= 0:
		raise ValueError("batch_size must be greater than 0")

	if num_workers is None:
		num_workers = min(4, os.cpu_count() or 1)

	if pin_memory is None:
		pin_memory = torch.cuda.is_available()

	if persistent_workers is None:
		persistent_workers = num_workers > 0

	generator = None

	if seed is not None:
		generator = torch.Generator()
		generator.manual_seed(seed)

	train_data = datasets.ImageFolder(
		root=train_dir,
		transform=train_transform
	)

	test_data = datasets.ImageFolder(
		root=test_dir,
		transform=test_transform
	)

	class_names = train_data.classes
	class_to_idx = train_data.class_to_idx

	if train_data.class_to_idx != test_data.class_to_idx:
		raise ValueError(
			"Train and test datasets have different class_to_idx mappings"
			"Check that both folders contain the same class names"
		)

	train_dataloader = DataLoader(
		dataset=train_data,
		batch_size=batch_size,
		shuffle=shuffle_train,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=drop_last,
		generator=generator
	)

	test_dataloader = DataLoader(
		dataset=test_data,
		batch_size=batch_size,
		shuffle=shuffle_test,
		num_workers=num_workers,
		pin_memory=pin_memory,
		persistent_workers=persistent_workers,
		drop_last=False
	)

	return train_dataloader, test_dataloader, class_names, class_to_idx