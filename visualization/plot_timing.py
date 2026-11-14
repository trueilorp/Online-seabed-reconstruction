"""
Timing per seabed family in two log-scale panels: time per terrain on the full scan (Poisson, PPSurf), and
median RTF per window with the sliding window (Poisson + SW, PPSurf + SW) with the worst window as a dash and
the RTF = 1 line. Reads metrics_per_family.csv of the four runs.

Usage (from the repository root):
	python visualization/plot_timing.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--out imgs/timing_per_family.png
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
FULL = [
	("poisson", "Poisson", BLUE, BLUE, None),
	("ppsurf", "PPSurf", ORANGE, ORANGE, None),
]
SW = [
	("poisson_sw", "Poisson + SW", LIGHT_BLUE, BLUE, "////"),
	("ppsurf_sw", "PPSurf + SW", LIGHT_ORANGE, ORANGE, "////"),
]


def read(run: str) -> pd.DataFrame:
	return pd.read_csv(os.path.join(run, "metrics_per_family.csv")).set_index("family")


def style_axis(ax, ticks, y_label):
	ax.set_yscale("log")
	ax.set_ylim(ticks[0] * 0.7, ticks[-1] * 1.4)
	ax.yaxis.set_major_locator(FixedLocator(ticks))
	ax.yaxis.set_minor_locator(NullLocator())
	ax.set_yticklabels([f"{t:g}" for t in ticks])
	ax.set_ylabel(y_label, color=INK_2)
	ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)
	ax.tick_params(axis="x", labelrotation=25)
	ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))


def bars(ax, data, series, families, column, max_column=None):
	x = np.arange(len(families))
	width, gap = 0.36, 0.04
	for i, (name, label, color, edge, hatch) in enumerate(series):
		d = data[name].reindex(families)
		pos = x + (i - 0.5) * (width + gap)
		plt.rcParams["hatch.color"] = edge
		ax.bar(pos, d[column], width, label=label, color=color, edgecolor=edge,
			   linewidth=0.8, hatch=hatch, zorder=2)
		if max_column:
			ax.scatter(pos, d[max_column], marker="_", s=120, color=INK, linewidths=1.5, zorder=3,
					   label="max" if i == len(series) - 1 else None)
	ax.set_xticks(x, families)


def main():
	parser = argparse.ArgumentParser(description="Timing per family, 4 configurations")
	for name, *_ in FULL + SW:
		parser.add_argument(f"--{name}", required=True, help="run folder")
	parser.add_argument("--out", default="imgs/timing_per_family.png")
	args = parser.parse_args()

	data = {name: read(getattr(args, name)) for name, *_ in FULL + SW}
	families = list(data["poisson"].index)

	plt.rcParams.update({"font.size": 10, "hatch.linewidth": 0.8})
	fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)

	bars(left, data, FULL, families, "compute_time_s")
	style_axis(left, [1, 2, 5, 10, 20, 50], "time per terrain [s]")
	left.set_title("Full scan", fontsize=11, pad=28)

	bars(right, data, SW, families, "median_rtf", "max_rtf")
	right.axhline(1, color=INK_2, linestyle="--", linewidth=1, zorder=1, label="RTF = 1")
	style_axis(right, [0.05, 0.1, 0.2, 0.5, 1, 2, 5], "RTF per window (lower is better)")
	handles, labels = right.get_legend_handles_labels()
	order = sorted(range(len(labels)), key=lambda i: labels[i] in ("max", "RTF = 1"))  # bars first
	right.legend([handles[i] for i in order], [labels[i] for i in order], ncols=4, frameon=False, fontsize=9,
				 loc="lower center", bbox_to_anchor=(0.5, 1.0))
	right.set_title("Sliding window", fontsize=11, pad=28)

	fig.tight_layout()
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
