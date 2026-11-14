"""
Precision, recall and F-score as a function of tau for the four configurations Poisson / PPSurf, with and
without SW. Line = mean across families, band = min-max across families. Reads metrics_tau_per_family.csv
(written by evaluate.py --recompute) of the four runs.

Usage (from the repository root):
	python visualization/plot_tau.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--out imgs/metrics_tau.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"      # Poisson, PPSurf
INK_2, GRID = "#5f5e58", "#e4e3de"

# (argument, label, colour, line style, marker)
SERIES = [
	("poisson", "Poisson", BLUE, "-", "o"),
	("poisson_sw", "Poisson + SW", BLUE, "--", "s"),
	("ppsurf", "PPSurf", ORANGE, "-", "o"),
	("ppsurf_sw", "PPSurf + SW", ORANGE, "--", "s"),
]
METRICS = [("precision", "Precision"), ("recall", "Recall"), ("fscore", "F-score")]


def read(run: str) -> pd.DataFrame:
	"""For each tau: mean, min and max across families of each metric."""
	df = pd.read_csv(os.path.join(run, "metrics_tau_per_family.csv"))
	return df.groupby("tau")[[m for m, _ in METRICS]].agg(["mean", "min", "max"])


def main():
	parser = argparse.ArgumentParser(description="Precision, recall and F-score as a function of tau")
	for name, *_ in SERIES:
		parser.add_argument(f"--{name}", required=True, help="run folder")
	parser.add_argument("--out", default="imgs/metrics_tau.png")
	args = parser.parse_args()

	data = {name: read(getattr(args, name)) for name, *_ in SERIES}

	plt.rcParams.update({"font.size": 10})
	fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150, sharey=True)
	for ax, (metric, title) in zip(axes, METRICS):
		for name, label, color, style, marker in SERIES:
			d = data[name][metric]
			tau = d.index.values
			ax.fill_between(tau, d["min"], d["max"], color=color, alpha=0.12, linewidth=0, zorder=1)
			ax.plot(tau, d["mean"], style, color=color, marker=marker, markersize=5,
					linewidth=2, label=label, zorder=2)
		ax.set_title(title, fontsize=11)
		ax.set_xlabel("τ [m]", color=INK_2)
		ax.set_ylim(0, 1.02)
		ax.grid(color=GRID, linewidth=0.8, zorder=0)
		for side in ("top", "right"):
			ax.spines[side].set_visible(False)
		for side in ("left", "bottom"):
			ax.spines[side].set_color(INK_2)
		ax.tick_params(colors=INK_2)
	for ax in axes:
		ax.set_xticks(sorted(data["poisson"].index.values))
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=(0, 0, 1, 0.92))

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
