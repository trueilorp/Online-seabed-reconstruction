"""
Report figures of an online reconstruction (online_reconstruction.py), in the style of the other report plots and
with the content and colours of video/render_video.py:
  online_quality.png   F-score at two tau and Chamfer per update, against the time of the newest data
  online_timing.png    latency per update split by phase (updates.csv t_* columns) against the budget
  online_plan.png      plan view of the final mesh: GT hillshade, area seen by the sensor, area covered by the mesh,
                       seen but not covered (the missing recall), boat track (centre of each ping footprint)
and, for the report, report/tables/online.tex (summary table) and report/tables/online_values.tex (the numbers
quoted in the text, as LaTeX macros), so figures, table and text always come from the same run.
Reads <run>/updates.csv, <run>/metrics_online.csv (evaluate_online.py --tau 0.2 0.5), the mesh snapshots, the
Unity CSV of <run>/config.txt and the GT patch.

Usage (from the repository root):
	python visualization/plot_online.py output/online/<run>
	python visualization/plot_online.py output/online/<run> --out-dir <folder>
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_online import coverage_grid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video"))
from render_video import ready_sim_times  # same mesh delay as the video

BLUE, PURPLE, TEAL = "#2a78d6", "#7b5cd6", "#1a9c86"     # F @ tau1, F @ tau2, Chamfer (as in the video)
INK, INK_2, GRID = "#1f1f1e", "#5f5e58", "#e4e3de"
LIGHT_GREY = "#d6d4ce"
AREA, HOLE = "#4fa3d9", "#eb6834"
PHASES = [("t_poisson", "Poisson", BLUE), ("t_crop", "density trim, cleanup, crop", "#9ec5f4"),
		  ("other", "voxel map, filter, normals, merge, write", LIGHT_GREY)]


def style(ax, y_label=None, x_label=None):
	if y_label:
		ax.set_ylabel(y_label, color=INK_2)
	if x_label:
		ax.set_xlabel(x_label, color=INK_2)
	ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	for side in ("left", "bottom"):
		ax.spines[side].set_color(INK_2)
	ax.tick_params(colors=INK_2)


def full_batches(up: pd.DataFrame, every: int) -> pd.Series:
	return (up["last_ping"] - up["first_ping"] + 1) >= every


def plot_quality(up, metrics, taus, out):
	fig, (ax_f, ax_c) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True, height_ratios=[1.3, 1])
	for tau, color, ls in zip(taus, [BLUE, PURPLE], ["-", "--"]):
		m = metrics[np.isclose(metrics["tau"], tau)].set_index("update")["fscore"].reindex(up["update"])
		ax_f.plot(up["elapsed_s"], m.values, color=color, ls=ls, lw=2, marker="o", ms=4, label=f"F-score @ {tau:g} m",
				  zorder=3)
		ax_f.annotate(f"{m.values[-1]:.3f}", (up["elapsed_s"].iloc[-1], m.values[-1]), xytext=(6, 0),
					  textcoords="offset points", va="center", color=INK, fontsize=9)
	chamfer = metrics.drop_duplicates("update").set_index("update")["chamfer"].reindex(up["update"])
	ax_c.plot(up["elapsed_s"], chamfer.values, color=TEAL, lw=2, marker="o", ms=4, zorder=3)
	ax_c.annotate(f"{chamfer.values[-1]:.3f} m", (up["elapsed_s"].iloc[-1], chamfer.values[-1]), xytext=(6, 0),
				  textcoords="offset points", va="center", color=INK, fontsize=9)
	f_min = min(metrics.loc[metrics["tau"].isin(taus), "fscore"].min(), 1)
	ax_f.set_ylim(np.floor(f_min * 20) / 20 - 0.05, 1.0)
	ax_c.set_ylim(0, chamfer.max() * 1.25)
	style(ax_f, "F-score")
	style(ax_c, "Chamfer [m]", "simulation time of the newest data [s]")
	ax_f.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
	ax_c.set_xlim(0, up["elapsed_s"].max() * 1.08)
	fig.tight_layout()
	fig.savefig(out, dpi=200)
	plt.close(fig)


def plot_timing(up, every, out):
	full = full_batches(up, every)
	d = up[full].copy()
	d["other"] = d["latency_s"] - d["t_poisson"] - d["t_crop"]  # read/parse are outside the latency
	fig, ax = plt.subplots(figsize=(7.5, 3.4))
	bottom = np.zeros(len(d))
	for col, label, color in PHASES:
		ax.bar(d["update"], d[col], 0.7, bottom=bottom, color=color, edgecolor="white", linewidth=0.6, label=label,
			   zorder=2)
		bottom += d[col].to_numpy()
	budget = float(d["budget_s"].median())
	ax.axhline(budget, color=INK_2, ls="--", lw=1, zorder=3)
	ax.annotate(f"budget {budget:.0f} s ({every} pings)", (d["update"].iloc[0] - 0.4, budget), xytext=(0, 3),
				textcoords="offset points", va="bottom", color=INK_2, fontsize=9)
	for u, lat, rtf in zip(d["update"], d["latency_s"], d["rtf"]):
		ax.annotate(f"{rtf:.2f}", (u, lat), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
					color=INK_2, fontsize=7.5)
	ax.set_ylim(0, max(budget, d["latency_s"].max()) * 1.25)
	ax.set_xticks(d["update"])
	style(ax, "latency [s]", "update (label: RTF)")
	ax.legend(ncols=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8.5)
	fig.tight_layout()
	fig.savefig(out, dpi=200)
	plt.close(fig)
	if (~full).any():
		print(f"  online_timing: final partial batch (update {int(up.loc[~full, 'update'].iloc[-1])}) not shown")


def plot_plan(run, scan, gt_dir, terrain_xz, radius, out):
	meta = json.load(open(os.path.join(gt_dir, "meta.json")))
	cell, n = meta["cell_m"], meta["samples"]
	gt = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "gt_mesh.ply"))
	elevation = np.asarray(gt.vertices)[:, 2].reshape(n, n)
	valid = np.load(os.path.join(gt_dir, "valid_mask.npy"))
	px, pz = terrain_xz
	origin = (px - cell / 2, pz - cell / 2)
	extent = [origin[0], origin[0] + n * cell, origin[1], origin[1] + n * cell]

	seen =coverage_grid(scan["x"].to_numpy(), scan["z"].to_numpy(), origin[0], origin[1], cell, n, radius) & valid
	paths = sorted(glob.glob(os.path.join(run, "meshes", "mesh_*.ply"))) or [os.path.join(run, "mesh.ply")]
	mesh = o3d.io.read_triangle_mesh(paths[-1])
	c = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)
	cols = np.floor((c[:, 0] - origin[0]) / cell).astype(int)
	rows = np.floor((c[:, 2] - origin[1]) / cell).astype(int)
	ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
	covered = np.zeros((n, n), dtype=bool)
	covered[rows[ok], cols[ok]] = True
	holes = seen & ~covered
	g = scan.groupby("ping_number")
	track = np.c_[(g["x"].min() + g["x"].max()) / 2, (g["z"].min() + g["z"].max()) / 2]

	shade = LightSource(azdeg=315, altdeg=45).hillshade(elevation, vert_exag=3, dx=cell, dy=cell)
	fig, ax = plt.subplots(figsize=(7.5, 5.2))
	ax.imshow(np.where(valid, shade, np.nan), origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1.15,
			  interpolation="bilinear")
	ax.imshow(np.where(covered, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([AREA]), alpha=0.45,
			  interpolation="nearest")
	ax.imshow(np.where(holes, 1.0, np.nan), origin="lower", extent=extent, cmap=ListedColormap([HOLE]), alpha=0.9,
			  interpolation="nearest")
	ax.contour(seen.astype(float), levels=[0.5], colors=[INK], linewidths=1.0, origin="lower", extent=extent)
	ax.plot(track[:, 0], track[:, 1], color=INK, lw=1.4)
	ax.plot(*track[-1], "o", color=INK, ms=5)
	rr, cc = np.nonzero(seen)
	m = 12
	ax.set_xlim(origin[0] + cc.min() * cell - m, origin[0] + (cc.max() + 1) * cell + m)
	ax.set_ylim(origin[1] + rr.min() * cell - m, origin[1] + (rr.max() + 1) * cell + m)
	ax.set_aspect("equal")
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(colors=INK_2)
	ax.set_xlabel("x [m] (east)", color=INK_2); ax.set_ylabel("z [m] (north)", color=INK_2)
	pct = 100 * holes.sum() / max(seen.sum(), 1)
	handles = [Line2D([], [], color=INK, lw=1.0, label=f"area seen by the sensor (within {radius:g} m of a point)"),
			   Patch(facecolor=AREA, alpha=0.45, label="covered by the final mesh"),
			   Patch(facecolor=HOLE, alpha=0.9, label=f"seen but not covered ({pct:.1f}% of the seen area)"),
			   Line2D([], [], color=INK, lw=1.4, marker="o", markevery=[1], ms=5,
					  label="boat track (footprint centre), end")]
	ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2, fontsize=8.5,
			  labelcolor=INK)
	fig.savefig(out, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"  online_plan: seen {seen.sum() * cell ** 2:.0f} m2, not covered {holes.sum() * cell ** 2:.0f} m2 ({pct:.1f}%)")
	return {"seen_m2": seen.sum() * cell ** 2, "holes_pct": pct}


def write_tex(up, metrics, taus, every, scan, plan, run, tables_dir):
	"""tables/online.tex (summary table) and tables/online_values.tex (numbers quoted in the text, as macros)."""
	full = full_batches(up, every)
	ping_t = scan.groupby("ping_number")["elapsed_s"].max().sort_index().to_numpy()
	ping_dt = float(np.median(np.diff(ping_t))) if len(ping_t) > 1 else 1.0
	ready, rate = ready_sim_times(up, ping_dt, full)
	up = up.assign(mesh_delay_s=ready - up["elapsed_s"])
	wide = metrics.pivot_table(index="update", columns="tau", values=["precision", "recall", "fscore"])
	chamfer = metrics.drop_duplicates("update").set_index("update")["chamfer"]
	median_dist = metrics.drop_duplicates("update").set_index("update")["rec2gt_median"]
	last, first = wide.index.max(), wide.index.min()
	timing = up[full]

	def col(m, tau):
		return wide[(m, wide.columns.get_level_values(1)[np.isclose(wide.columns.get_level_values(1), tau)][0])]

	rows = []
	for tau in taus:
		for m, name in (("precision", "Precision"), ("recall", "Recall"), ("fscore", "F-score")):
			c = col(m, tau)
			rows.append(rf"{name} @ {tau:g}\,m & {c.mean():.3f} $\pm$ {c.std():.3f} & {c[last]:.3f} \\")
	rows.append(rf"Chamfer [m] & {chamfer.mean():.3f} $\pm$ {chamfer.std():.3f} & {chamfer[last]:.3f} \\")
	table = [r"\begin{tabular}{lcc}", r"\toprule", r"Metric & mean $\pm$ std over updates & final mesh \\",
			 r"\midrule", *rows, r"\midrule", r" & median & max \\", r"\midrule",
			 rf"Latency [s] & {timing['latency_s'].median():.2f} & {timing['latency_s'].max():.2f} \\",
			 rf"RTF & {timing['rtf'].median():.2f} & {timing['rtf'].max():.2f} \\",
			 rf"Mesh delay [s] & {timing['mesh_delay_s'].median():.2f} & {timing['mesh_delay_s'].max():.2f} \\",
			 r"\bottomrule", r"\end{tabular}", ""]
	share = (timing["t_poisson"] / timing["latency_s"]).median() if "t_poisson" in timing else float("nan")
	values = {
		"onRun": os.path.basename(os.path.normpath(run)).replace("_", r"\_"),
		"onUpdates": f"{len(up)}", "onFullUpdates": f"{len(timing)}", "onPings": f"{scan['ping_number'].nunique()}",
		"onPoints": f"{len(scan):,}".replace(",", "\\,"), "onDuration": f"{scan['elapsed_s'].max():.0f}",
		"onPointsPerPing": f"{len(scan) / scan['ping_number'].nunique() / 1000:.0f}",
		"onEvery": f"{every}", "onSimRate": f"{rate:.2f}",
		"onArea": f"{plan['seen_m2']:,.0f}".replace(",", "\\,"), "onHoles": f"{plan['holes_pct']:.0f}",
		"onPfive": f"{col('precision', 0.5)[last]:.3f}", "onRfive": f"{col('recall', 0.5)[last]:.3f}",
		"onFfive": f"{col('fscore', 0.5)[last]:.3f}", "onFtwo": f"{col('fscore', 0.2)[last]:.3f}",
		"onRfiveFirst": f"{col('recall', 0.5)[first]:.3f}", "onFfiveFirst": f"{col('fscore', 0.5)[first]:.3f}",
		"onChamfer": f"{chamfer[last]:.2f}", "onChamferFirst": f"{chamfer[first]:.2f}",
		"onMedianDist": f"{100 * median_dist[last]:.0f}",
		"onLatMed": f"{timing['latency_s'].median():.1f}", "onLatMax": f"{timing['latency_s'].max():.1f}",
		"onRtfMed": f"{timing['rtf'].median():.2f}", "onRtfMax": f"{timing['rtf'].max():.2f}",
		"onInTime": ("all" if (timing["rtf"] <= 1).all() else f"{int((timing['rtf'] <= 1).sum())} of")
					+ f" {len(timing)}",
		"onDelayMed": f"{timing['mesh_delay_s'].median():.1f}", "onPoissonShare": f"{100 * share:.0f}",
	}
	os.makedirs(tables_dir, exist_ok=True)
	with open(os.path.join(tables_dir, "online.tex"), "w") as f:
		f.write("\n".join(table))
	with open(os.path.join(tables_dir, "online_values.tex"), "w") as f:
		f.write(f"% generated by visualization/plot_online.py from {run}\n")
		f.writelines(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in values.items())
	print(f"  wrote {tables_dir}/online.tex and online_values.tex")


def main():
	parser = argparse.ArgumentParser(description="Report figures of an online reconstruction run")
	parser.add_argument("run", help="online_reconstruction.py run folder (with metrics_online.csv)")
	parser.add_argument("--taus", type=float, nargs=2, default=[0.2, 0.5])
	parser.add_argument("--gt", default="data/real/patch_malamocco")
	parser.add_argument("--terrain-position", type=float, nargs=2, default=[-128.0, -128.0], metavar=("X", "Z"))
	parser.add_argument("--coverage-radius", type=float, default=1.0, help="as evaluate_online.py")
	parser.add_argument("--out-dir", default="imgs")
	parser.add_argument("--tables-dir", default="report/tables", help="where online.tex and online_values.tex go")
	a = parser.parse_args()

	config = yaml.safe_load(open(os.path.join(a.run, "config.txt")))
	every = int(config["online"]["update_every_pings"])
	up = pd.read_csv(os.path.join(a.run, "updates.csv"))
	snapshots = {int(re.search(r"mesh_(\d+)\.ply$", f).group(1)) for f in glob.glob(os.path.join(a.run, "meshes", "*.ply"))}
	metrics_path = os.path.join(a.run, "metrics_online.csv")
	if not os.path.exists(metrics_path):
		raise SystemExit(f"No metrics_online.csv: run python evaluate_online.py {a.run} --tau 0.2 0.5")
	metrics = pd.read_csv(metrics_path)
	for tau in a.taus:
		if not np.isclose(metrics["tau"], tau).any():
			raise SystemExit(f"metrics_online.csv has no tau = {tau:g}: run evaluate_online.py with --tau 0.2 0.5")
	os.makedirs(a.out_dir, exist_ok=True)

	plot_quality(up[up["update"].isin(snapshots)], metrics, a.taus, os.path.join(a.out_dir, "online_quality.png"))
	plot_timing(up, every, os.path.join(a.out_dir, "online_timing.png"))
	scan = pd.read_csv(config["csv"], usecols=["x", "z", "ping_number", "elapsed_s"])
	plan = plot_plan(a.run, scan, a.gt, a.terrain_position, a.coverage_radius,
					 os.path.join(a.out_dir, "online_plan.png"))
	write_tex(up, metrics, a.taus, every, scan, plan, a.run, a.tables_dir)
	print(f"Saved the figures in {a.out_dir}/")


if __name__ == "__main__":
	main()
