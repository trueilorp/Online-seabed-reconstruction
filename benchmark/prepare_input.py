"""Runs the shared preprocessing once for the timing scripts (benchmark/time_*.sh): saves the cleaned point
cloud to <out_dir>/input.ply and writes <out_dir>/params.sh with the method parameters from the config.
The sliding window is ignored.

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/prepare_input.py --method ppsurf --out_dir benchmark/benchmark_output/ppsurf/<dataset> --csv <id>_scan.csv
"""

import argparse
import os
import shlex
import sys
import time

import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import load_config, load_and_clean_point_cloud, run_name_from_config
from scripts.ppsurf import subsample_if_needed


def save_points_only(pcd: o3d.geometry.PointCloud, path: str):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	points_only = o3d.geometry.PointCloud()
	points_only.points = pcd.points
	o3d.io.write_point_cloud(path, points_only)


def ground_truth_path(csv: str) -> str:
	"""<id>_ground_truth.obj next to <id>_scan.csv (synthetic terrains), "" if missing."""
	folder, name = os.path.split(csv)
	terrain_id = os.path.splitext(name)[0].removesuffix("_scan")
	candidate = os.path.join(folder, f"{terrain_id}_ground_truth.obj")
	if os.path.isfile(candidate):
		return os.path.abspath(candidate)
	print("[prepare_input] No ground truth found: surface metrics not computed")
	return ""


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--method", required=True, choices=["poisson", "ppsurf"])
	parser.add_argument("--out_dir", required=True)
	parser.add_argument("--csv", required=True, help="scan CSV (<id>_scan.csv)")
	args = parser.parse_args()

	config = load_config()
	config["paths"]["input_csv"] = args.csv
	config["paths"].pop("run_name", None)
	run_name = run_name_from_config(config)
	out_dir = os.path.abspath(args.out_dir)
	os.makedirs(out_dir, exist_ok=True)

	start = time.time()
	pcd = load_and_clean_point_cloud(config)
	print(f"[prepare_input] Loading + cleaning: {time.time() - start:.2f}s, {len(pcd.points)} points")

	params = {"RUN_NAME": run_name, "GT_MESH": ground_truth_path(args.csv)}
	if args.method == "ppsurf":
		pps = config["ppsurf"]
		pcd = subsample_if_needed(pcd, pps["max_points"])
		params.update(
			CONDA_ENV=pps["conda_env"],
			METHOD_DIR=os.path.abspath(os.path.join(config["paths"]["neural_methods"], "ppsurf")),
			PPS_RES=pps["gen_resolution_global"],
			PPS_BATCH=pps["rec_batch_size"],
		)
	save_points_only(pcd, os.path.join(out_dir, "input.ply"))

	params["N_POINTS"] = len(pcd.points)
	with open(os.path.join(out_dir, "params.sh"), "w") as f:
		for key, value in params.items():
			f.write(f"{key}={shlex.quote(str(value))}\n")
	print(f"[prepare_input] Input ready in {out_dir} ({params['N_POINTS']} points for {args.method})")


if __name__ == "__main__":
	main()
