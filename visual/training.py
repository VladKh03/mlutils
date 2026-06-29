from pathlib import Path
from typing import Optional, Literal

import matplotlib.pyplot as plt


def _save_or_show_plot(
	save_path: Optional[str | Path] = None,
	show: bool = True,
	dpi: int = 150
) -> None:
	"""
	Saves and/or shows current matplotlib plot.
	"""

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches="tight", dpi=dpi)

	if show:
		plt.show()
	else:
		plt.close()


def plot_training_curves(
	results: dict,
	keys: list[str],
	title: str = "Training curves",
	xlabel: str = "Epoch",
	ylabel: str = "Value",
	figsize: tuple[int, int] = (10, 6),
	save_path: Optional[str | Path] = None,
	show: bool = True,
	grid: bool = True,
	dpi: int = 150
) -> None:
	"""
	Plots one or more curves from a training results dictionary.

	Args:
		results: Dictionary with training history.
		keys: Metric names from results to plot.
		title: Plot title.
		xlabel: X-axis label.
		ylabel: Y-axis label.
		figsize: Figure size.
		save_path: Optional path to save the plot.
		show: If True, shows the plot.
		grid: If True, shows grid.
		dpi: Image quality when saving.
	"""

	if len(keys) == 0:
		raise ValueError("keys cannot be empty")

	for key in keys:
		if key not in results:
			raise KeyError(f"results does not contain '{key}'")

	first_key = keys[0]
	epochs = range(1, len(results[first_key]) + 1)

	plt.figure(figsize=figsize)

	for key in keys:
		values = results[key]

		if len(values) != len(results[first_key]):
			raise ValueError(f"'{key}' has different length than '{first_key}'")

		plt.plot(
			epochs,
			values,
			label=key
		)

	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()

	if grid:
		plt.grid(True, alpha=0.3)

	_save_or_show_plot(
		save_path=save_path,
		show=show,
		dpi=dpi
	)