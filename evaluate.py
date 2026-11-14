"""
Evaluates the reconstruction methods on the synthetic terrains: reconstructs each <id>_scan.csv with
utils/pipeline.py and compares the mesh with the seabed of <id>_ground_truth.obj (Chamfer, precision,
recall, F-score at tau, distance statistics; latency and RTF per window with the sliding window).

Each run writes a new folder <evaluation.output_dir>/<date_time>_<split>[_sw]/ with config.txt, log.txt,
git_diff.patch, metrics_per_terrain.csv, metrics_per_family.csv, windows.csv (SW only), meshes/ and
distances/ (raw distances, from which --recompute derives the metrics at other thresholds).

Usage:
	python evaluate.py --split val --limit 2 --methods poisson ppsurf      # 2 terrains per family
	python evaluate.py --split test+val --sliding_window on
	python evaluate.py --recompute output/evaluation/<run> --tau 0.1 0.2 0.5
	python evaluate.py --methods ppsurf --set ppsurf.gen_refine_iter=5
"""

import argparse
import atexit
import copy
import datetime
import glob
import os
import sys
import traceback

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from utils.mesh import top_surface
from utils.run_info import environment, git_state, TeeLog, ResourceMonitor
from utils.pipeline import load_config, config_for_terrain, reconstruct, RUNNERS


def distances_to_mesh(points: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
	"""Exact point-to-surface distance (not point-to-point) via RaycastingScene."""
	scene = o3d.t.geometry.RaycastingScene()
	scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def surface_metrics(d_rec: np.ndarray, d_seabed: np.ndarray, tau: float) -> dict:
	"""Chamfer and F-score against the seabed. d_rec: distances of the reconstruction points from the seabed;
	d_seabed: distances of the seabed points from the reconstruction."""
	precision = float((d_rec <= tau).mean())
	recall = float((d_seabed <= tau).mean())
	return {
		"chamfer": float((d_rec.mean() + d_seabed.mean()) / 2),
		"precision": precision,
		"recall": recall,
		"fscore": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
	}


def distance_stats(d: np.ndarray, prefix: str) -> dict:
	"""Median, interquartile range and 95th percentile of a distance vector."""
	p25, p50, p75, p95 = np.percentile(d, [25, 50, 75, 95])
	return {f"{prefix}_median": float(p50), f"{prefix}_iqr": float(p75 - p25), f"{prefix}_p95": float(p95)}


def window_summary(windows: list) -> dict:
	"""Per-terrain columns from the window statistics (sliding window)."""
	ok = [w for w in windows if w["status"] == "ok"]
	if not ok:
		return {}
	latencies = np.array([w["latency_s"] for w in ok])
	rtf = np.array([w["rtf"] for w in ok if w["rtf"] is not None])
	summary = {"n_windows": len(ok), "median_latency_s": float(np.median(latencies)),
			   "max_latency_s": float(latencies.max())}
	if len(rtf):
		summary.update(median_rtf=float(np.median(rtf)), max_rtf=float(rtf.max()),
					   windows_in_time=float((rtf <= 1).mean()))
	return summary


def compute_metrics(mesh: o3d.geometry.TriangleMesh, seabed: o3d.geometry.TriangleMesh, tau: float, n_samples: int):
	"""Aggregate metrics + raw distances {"rec2gt": reconstruction->seabed, "gt2rec": seabed->reconstruction}."""
	mesh_points = np.asarray(mesh.sample_points_uniformly(n_samples).points)
	seabed_points = np.asarray(seabed.sample_points_uniformly(n_samples).points)

	rec2gt = distances_to_mesh(mesh_points, seabed)
	gt2rec = distances_to_mesh(seabed_points, mesh)
	metrics = {
		**surface_metrics(rec2gt, gt2rec, tau),
		**distance_stats(rec2gt, "rec2gt"),
		**distance_stats(gt2rec, "gt2rec"),
	}
	return metrics, {"rec2gt": rec2gt, "gt2rec": gt2rec}


def list_terrains(synthetic_dir: str, split: str, families: dict, limit: int = None) -> list:
	"""Terrains of the split in alphabetical order; with limit, the first limit of EACH family."""
	folder = os.path.join(synthetic_dir, split)
	ids = sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)))
	if not limit:
		return ids
	per_family = {}
	for terrain_id in ids:
		per_family.setdefault(family_of(terrain_id, families), []).append(terrain_id)
	return [i for group in per_family.values() for i in group[:limit]]


def family_of(terrain_id: str, families: dict) -> str:
	"""Family from the manifest, otherwise from the id prefix (flat_sand_012 -> flat_sand)."""
	return families.get(terrain_id, terrain_id.rsplit("_", 1)[0])


def families_from_manifest(synthetic_dir: str) -> dict:
	manifest = os.path.join(synthetic_dir, "manifest.csv")
	if not os.path.exists(manifest):
		return {}
	df = pd.read_csv(manifest)
	return dict(zip(df["id"], df["family"]))


CSV_TERRAIN = "metrics_per_terrain.csv"
CSV_FAMILY = "metrics_per_family.csv"
CSV_WINDOWS = "windows.csv"
QUALITY_COLUMNS = ["chamfer", "precision", "recall", "fscore"]


def run_folder(output_dir: str, split: str, sliding_window: bool) -> str:
	"""New folder for this run: <output_dir>/<date_time>_<split>[_sw]."""
	now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	return os.path.join(output_dir, f"{now}_{split}{'_sw' if sliding_window else ''}")


def latest_run(output_dir: str, split: str, sliding_window: bool = None) -> str:
	"""Most recent run folder of a split (for the benchmark scripts that read its meshes).
	sliding_window: True/False to select runs with/without SW, None = any."""
	suffixes = {None: ["", "_sw"], False: [""], True: ["_sw"]}[sliding_window]
	folders = sorted(c for s in suffixes for c in glob.glob(os.path.join(output_dir, f"*_{split}{s}")))
	if not folders:
		raise FileNotFoundError(f"No evaluate.py run for split {split} in {output_dir}")
	return folders[-1]  # names start with date and time: alphabetical order is chronological


def save_config(config: dict, out_folder: str):
	"""config.txt: command, effective config (YAML), machine and code version; git_diff.patch if uncommitted."""
	ppsurf_python = None
	if "ppsurf" in config["evaluation"]["methods"]:
		try:
			from scripts.ppsurf import find_conda_env_python
			ppsurf_python = find_conda_env_python(config["ppsurf"]["conda_env"], "")
		except Exception:
			pass
	git, diff = git_state()
	with open(os.path.join(out_folder, "config.txt"), "w") as f:
		f.write(f"# command: python {' '.join(sys.argv)}\n")
		f.write(f"# started: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
		f.write("# effective config (config/config.yaml + command-line options):\n\n")
		yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
		f.write("\n# ---- machine and versions ----\n")
		yaml.safe_dump({"environment": environment(ppsurf_python)}, f, sort_keys=False, allow_unicode=True)
		f.write("\n# ---- code ----\n")
		yaml.safe_dump({"git": git}, f, sort_keys=False, allow_unicode=True)
	if diff:
		with open(os.path.join(out_folder, "git_diff.patch"), "w") as f:
			f.write(diff + "\n")


def save_csv(rows: list, out_folder: str):
	"""Per-terrain CSV and per-family means of this run."""
	df = pd.DataFrame(rows)
	df.to_csv(os.path.join(out_folder, CSV_TERRAIN), index=False, float_format="%.3f")

	ok = df[df["error"].isna()] if "error" in df else df
	if ok.empty:
		return
	columns = ["chamfer", "precision", "recall", "fscore",
			   "rec2gt_median", "rec2gt_iqr", "rec2gt_p95", "gt2rec_median", "gt2rec_iqr", "gt2rec_p95",
			   "n_input_points", "n_vertices", "compute_time_s", "peak_ram_mb", "peak_vram_mb",
			   "median_latency_s", "max_latency_s", "median_rtf", "max_rtf", "windows_in_time"]
	columns = [c for c in columns if c in ok.columns]  # the SW columns exist only with SW
	groups = ["family", "method", "sliding_window"]
	mean_and_std(ok, groups, columns).to_csv(os.path.join(out_folder, CSV_FAMILY), index=False, float_format="%.3f")


def mean_and_std(df: pd.DataFrame, groups: list, columns: list) -> pd.DataFrame:
	"""Per-group means, with <col>_std (standard deviation across terrains) right after the quality metrics.
	The std is empty with a single terrain in the group."""
	g = df.groupby(groups)
	means = g[columns].mean()
	std = g[[c for c in QUALITY_COLUMNS if c in columns]].std()
	order = []
	for c in columns:
		order.append(c)
		if c in std.columns:
			means[f"{c}_std"] = std[c]
			order.append(f"{c}_std")
	means = means[order]
	means.insert(0, "n_terrains", g.size())
	return means.reset_index()


def save_windows(windows: list, out_folder: str, terrain_id: str, method: str):
	"""Appends the per-window rows to windows.csv (one write per terrain: a crash keeps the previous ones)."""
	df = pd.DataFrame(windows)
	df.insert(0, "id", terrain_id)
	df.insert(1, "method", method)
	path = os.path.join(out_folder, CSV_WINDOWS)
	df.to_csv(path, mode="a", header=not os.path.exists(path), index=False, float_format="%.3f")


def recompute_tau(run: str, tau: list):
	"""Metrics at other thresholds tau from the raw distances of a run (no reconstruction)."""
	per_terrain = pd.read_csv(os.path.join(run, CSV_TERRAIN))
	if "error" in per_terrain:
		per_terrain = per_terrain[per_terrain["error"].isna()]
	rows = []
	for _, r in per_terrain.iterrows():
		suffix = "_sw" if str(r["sliding_window"]).lower() == "true" else ""
		base = os.path.join(run, "distances", f"{r['id']}_{r['method']}{suffix}")
		if not os.path.exists(f"{base}_rec2gt.npy"):
			print(f"  missing distances for {r['id']} {r['method']}: skipped")
			continue
		d_rec, d_gt = np.load(f"{base}_rec2gt.npy"), np.load(f"{base}_gt2rec.npy")
		for t in tau:
			rows.append({"id": r["id"], "family": r["family"], "method": r["method"],
						 "sliding_window": r["sliding_window"], "tau": t, **surface_metrics(d_rec, d_gt, t)})
	if not rows:
		sys.exit(f"No distances in {run}/distances")
	df = pd.DataFrame(rows)
	df.to_csv(os.path.join(run, "metrics_tau.csv"), index=False, float_format="%.3f")
	groups = ["family", "method", "sliding_window", "tau"]
	mean_and_std(df, groups, QUALITY_COLUMNS).to_csv(
		os.path.join(run, "metrics_tau_per_family.csv"), index=False, float_format="%.3f")
	overall = df.groupby(["method", "sliding_window", "tau"])[QUALITY_COLUMNS].mean()
	print(overall.round(3).to_string())
	print(f"\nWrote metrics_tau.csv and metrics_tau_per_family.csv in {run}")


def apply_overrides(config: dict, assignments: list, parser: argparse.ArgumentParser):
	"""--set SECTION.KEY=VALUE: values parsed as YAML (5 -> int, 0.5 -> float, null -> None, true -> bool)."""
	for assignment in assignments:
		key, sep, value = assignment.partition("=")
		if not sep:
			parser.error(f"--set expects SECTION.KEY=VALUE, not '{assignment}'")
		*sections, last = key.split(".")
		node = config
		for section in sections:
			if not isinstance(node.get(section), dict):
				parser.error(f"--set: section '{section}' not in the config")
			node = node[section]
		if last not in node:
			parser.error(f"--set: key '{key}' not in the config")
		node[last] = yaml.safe_load(value)


def main():
	base_config = load_config()
	ev = base_config["evaluation"]

	parser = argparse.ArgumentParser(description="Evaluates the reconstruction methods on the synthetic terrains.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--limit", type=int, default=ev["limit"], help="maximum number of terrains per family")
	parser.add_argument("--methods", nargs="+", default=ev["methods"], choices=list(RUNNERS))
	parser.add_argument("--tau", type=float, nargs="+", default=[ev["tau"]],
						help="threshold in metres; several values only with --recompute")
	parser.add_argument("--recompute", metavar="RUN_FOLDER", default=None,
						help="no reconstruction: recompute the metrics for each --tau from the distances of this run")
	parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="SECTION.KEY=VALUE",
						help="overrides config values, e.g. --set ppsurf.gen_refine_iter=5 poisson.depth=8")
	parser.add_argument("--sliding_window", choices=["on", "off"], default=None,
						help="overrides sliding_window.enabled of the config")
	args = parser.parse_args()

	if args.recompute:
		recompute_tau(args.recompute, args.tau)
		return
	if len(args.tau) > 1:
		parser.error("several --tau values only with --recompute")
	args.tau = args.tau[0]

	# command-line options go into the config, so config.txt describes the run actually executed
	base_config = copy.deepcopy(base_config)
	apply_overrides(base_config, args.set, parser)
	if args.sliding_window is not None:
		base_config.setdefault("sliding_window", {})["enabled"] = args.sliding_window == "on"
	ev = base_config["evaluation"]
	ev.update(split=args.split, limit=args.limit, methods=args.methods, tau=args.tau)
	sliding_window = bool(base_config.get("sliding_window", {}).get("enabled"))
	suffix = "_sw" if sliding_window else ""

	out_folder = run_folder(ev["output_dir"], args.split, sliding_window)
	mesh_folder = os.path.join(out_folder, "meshes")
	distance_folder = os.path.join(out_folder, "distances")
	os.makedirs(mesh_folder, exist_ok=True)
	os.makedirs(distance_folder, exist_ok=True)
	tee = TeeLog(os.path.join(out_folder, "log.txt")).__enter__()
	atexit.register(tee.__exit__, None, None, None)
	print(f"Run in: {out_folder}")
	save_config(base_config, out_folder)

	families = families_from_manifest(ev["synthetic_dir"])
	terrains = list_terrains(ev["synthetic_dir"], args.split, families, args.limit)
	rows = []

	# sequential: the neural methods share the work_dir
	for i, terrain_id in enumerate(terrains, 1):
		folder = os.path.join(ev["synthetic_dir"], args.split, terrain_id)
		scan_path = os.path.join(folder, f"{terrain_id}_scan.csv")
		seabed = top_surface(o3d.io.read_triangle_mesh(os.path.join(folder, f"{terrain_id}_ground_truth.obj")))
		family = family_of(terrain_id, families)

		for method in args.methods:
			print(f"\n=== [{i}/{len(terrains)}] {terrain_id} - {method} ===")
			row = {"id": terrain_id, "family": family, "method": method, "sliding_window": sliding_window}
			try:
				config = config_for_terrain(base_config, scan_path, method)
				windows = []
				with ResourceMonitor() as resources:
					mesh, pcd, elapsed, name = reconstruct(config, windows)
				row["method"] = name
				# points received by the method after preprocessing (with SW: sum over the windows,
				# overlapping pings count twice)
				row["n_input_points"] = (sum(w.get("n_clean_points") or 0 for w in windows) if windows
										 else len(pcd.points))
				row["peak_ram_mb"] = resources.peak_ram_mb
				row["peak_vram_mb"] = resources.peak_vram_mb

				mesh_out = os.path.join(mesh_folder, f"{terrain_id}_{name}{suffix}.ply")
				o3d.io.write_triangle_mesh(mesh_out, mesh)

				metrics, distances = compute_metrics(mesh, seabed, args.tau, ev["n_samples"])
				row.update(metrics)
				for direction, d in distances.items():
					np.save(os.path.join(distance_folder, f"{terrain_id}_{name}{suffix}_{direction}.npy"),
							d.astype(np.float32))
				row.update({"n_vertices": len(mesh.vertices), "compute_time_s": elapsed,
							"mesh": mesh_out, "error": None})
				if windows:
					row.update(window_summary(windows))
					save_windows(windows, out_folder, terrain_id, name)
				print("  " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float)))
			except Exception as e:
				traceback.print_exc()
				row["error"] = f"{type(e).__name__}: {e}"
			rows.append(row)
			save_csv(rows, out_folder)  # rewritten at every step: a crash does not lose results

	print(f"\nResults in: {out_folder}")


if __name__ == "__main__":
	main()
