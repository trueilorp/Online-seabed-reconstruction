"""
Real-time factor per window (sliding window), Poisson + SW against PPSurf + SW. Reads windows.csv of the two
SW runs; x = window index along the track, y = rtf = latency_s / budget_s (log scale). Line = median across
terrains, band = 10th-90th percentile; the dashed line at RTF = 1 separates windows processed in time.

Usage (from the repository root):
	python visualization/plot_rtf.py \
		--poisson_sw output/evaluation/<poisson_sw_run> \
		--ppsurf_sw  output/evaluation/<ppsurf_sw_run> \
		--out imgs/rtf_windows.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

BLUE, ORANGE = "#2a78d6", "#eb6834"      # Poisson, PPSurf
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"

SERIES = [("poisson_sw", "Poisson + SW", BLUE), ("ppsurf_sw", "PPSurf + SW", ORANGE)]


def read(run: str) -> pd.DataFrame:
	"""Median and 10th/90th percentiles of rtf across terrains, per window index."""
	df = pd.read_csv(os.path.join(run, "windows.csv"))
	df = df[(df["status"] == "ok") & df["rtf"].notna()]
	g = df.groupby("window")["rtf"]
	return pd.DataFrame({"median": g.median(), "p10": g.quantile(0.1), "p90": g.quantile(0.9), "n": g.size()})


def main():
	parser = argparse.ArgumentParser(description="RTF per window, Poisson + SW against PPSurf + SW")
	for name, *_ in SERIES:
		parser.add_argument(f"--{name}", required=True, help="run folder with windows.csv")
	parser.add_argument("--out", default="imgs/rtf_windows.png")
	args = parser.parse_args()

	plt.rcParams.update({"font.size": 10})
	fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
	for name, label, color in SERIES:
		d = read(getattr(args, name))
		ax.fill_between(d.index, d["p10"], d["p90"], color=color, alpha=0.15, linewidth=0, zorder=1)
		ax.plot(d.index, d["median"], color=color, linewidth=2, marker="o", markersize=4, label=label, zorder=2)
	ax.axhline(1.0, color=INK, linewidth=1, linestyle="--", zorder=3)
	ax.text(ax.get_xlim()[1], 1.0, " RTF = 1", va="center", ha="left", color=INK, fontsize=9)

	ax.set_yscale("log")
	ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
	ax.yaxis.set_major_locator(FixedLocator(ticks))
	ax.yaxis.set_minor_locator(NullLocator())
	ax.set_yticklabels([f"{t:g}" for t in ticks])
	ax.set_xlabel("window (acquisition order)", color=INK_2)
	ax.set_ylabel("RTF = latency / budget", color=INK_2)
	ax.grid(color=GRID, linewidth=0.8, zorder=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)
	ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Saved {args.out}")


if __name__ == "__main__":
	main()
