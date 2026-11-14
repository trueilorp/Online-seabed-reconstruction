"""
Screenshot grid: for each chosen terrain, ground truth and the four configurations (Poisson, Poisson + SW,
PPSurf, PPSurf + SW) side by side with the same camera, cropped to the common non-white area. Uses the
screenshots saved by visualization/screenshot_run.py.

Usage (from the repository root):
	python visualization/plot_mesh_comparison.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--terrains flat_sand_012 rocky_012 mixed_012 --out imgs/mesh_comparison.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# (argument or None for the GT, column title, file suffix)
COLUMNS = [
	(None, "Ground truth", "_ground_truth"),
	("poisson", "Poisson", "_poisson"),
	("poisson_sw", "Poisson + SW", "_poisson_sw"),
	("ppsurf", "PPSurf", "_ppsurf"),
	("ppsurf_sw", "PPSurf + SW", "_ppsurf_sw"),
]


def non_white_box(images: list, threshold: float = 0.98):
	"""Common bounding box of the non-white pixels of a list of same-size images."""
	mask = np.zeros(images[0].shape[:2], dtype=bool)
	for im in images:
		mask |= im[..., :3].min(axis=2) < threshold
	rows, cols = np.where(mask)
	return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def main():
	parser = argparse.ArgumentParser(description="Grid of GT + 4 configurations for some terrains")
	for name, *_ in COLUMNS[1:]:
		parser.add_argument(f"--{name}", required=True, help="run folder with screenshots/")
	parser.add_argument("--ground_truth", default="output/groundtruth", help="folder of the GT screenshots")
	parser.add_argument("--terrains", nargs="+", default=["flat_sand_012", "rocky_012", "mixed_012"])
	parser.add_argument("--out", default="imgs/mesh_comparison.png")
	args = parser.parse_args()

	fig, axes = plt.subplots(len(args.terrains), len(COLUMNS), figsize=(13, 2.3 * len(args.terrains)), dpi=150,
							 squeeze=False)
	for r, terrain in enumerate(args.terrains):
		images = []
		for name, _, suffix in COLUMNS:
			folder = args.ground_truth if name is None else os.path.join(getattr(args, name), "screenshots")
			images.append(mpimg.imread(os.path.join(folder, f"{terrain}{suffix}.png")))
		y0, y1, x0, x1 = non_white_box(images)
		for c, (im, (_, title, _)) in enumerate(zip(images, COLUMNS)):
			ax = axes[r, c]
			ax.imshow(im[y0:y1, x0:x1])
			ax.set_xticks([])
			ax.set_yticks([])
			for side in ax.spines.values():
				side.set_visible(False)
			if r == 0:
				ax.set_title(title, fontsize=11)
			if c == 0:
				ax.set_ylabel(terrain, fontsize=10)
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
