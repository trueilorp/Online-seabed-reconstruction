"""
Bar chart of the Chamfer distance per seabed family (mean ± std across terrains) for the four configurations
Poisson / PPSurf, with and without SW, on a log scale. Reads metrics_per_family.csv of the four runs.

Usage (from the repository root):
	python visualization/plot_chamfer.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--out imgs/chamfer_per_family.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

BLUE, ORANGE = "#2a78d6", "#eb6834"      # Poisson, PPSurf
LIGHT_BLUE, LIGHT_ORANGE = "#9ec5f4", "#f6b59a"
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"

# (argument, label, fill, edge, hatch)
SERIES = [
	("poisson", "Poisson", BLUE, BLUE, None),
	("poisson_sw", "Poisson + SW", LIGHT_BLUE, BLUE, "////"),
	("ppsurf", "PPSurf", ORANGE, ORANGE, None),
	("ppsurf_sw", "PPSurf + SW", LIGHT_ORANGE, ORANGE, "////"),
]


def read(run: str) -> pd.DataFrame:
	df = pd.read_csv(os.path.join(run, "metrics_per_family.csv"))
	return df.set_index("family")[["chamfer", "chamfer_std"]]


def main():
	parser = argparse.ArgumentParser(description="Chamfer per family, 4 configurations")
	for name, *_ in SERIES:
		parser.add_argument(f"--{name}", required=True, help="run folder")
	parser.add_argument("--out", default="imgs/chamfer_per_family.png")
	args = parser.parse_args()

	data = {name: read(getattr(args, name)) for name, *_ in SERIES}
	families = list(data["poisson"].index)
	x = np.arange(len(families))
	width, gap = 0.19, 0.02

	plt.rcParams.update({"font.size": 10, "hatch.linewidth": 0.8})
	fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
	for i, (name, label, color, edge, hatch) in enumerate(SERIES):
		d = data[name].reindex(families)
		pos = x + (i - 1.5) * (width + gap)
		plt.rcParams["hatch.color"] = edge
		ax.bar(pos, d["chamfer"], width, label=label, color=color, edgecolor=edge,
			   linewidth=0.8, hatch=hatch, zorder=2)
		ax.errorbar(pos, d["chamfer"], yerr=d["chamfer_std"].fillna(0), fmt="none",
					ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)

	ax.set_yscale("log")
	ax.set_ylim(0.02, 1)
	ticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1]
	ax.yaxis.set_major_locator(FixedLocator(ticks))
	ax.yaxis.set_minor_locator(NullLocator())
	ax.set_yticklabels([f"{t:g}" for t in ticks])
	ax.set_ylabel("Chamfer [m] (lower is better)", color=INK_2)
	ax.set_xticks(x, families)
	ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)
	ax.legend(ncols=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
