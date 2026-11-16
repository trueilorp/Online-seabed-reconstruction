"""
Metrics of an online reconstruction (online_reconstruction.py) against the real-seabed ground truth
(data/real/<patch>/gt_mesh.ply, made by venice_to_unity.py): Chamfer, precision, recall and F-score at tau,
with the same functions as evaluate.py, for every mesh snapshot of the run (meshes/) or for mesh.ply only.

Frames: the GT is in local ENU (x east, y north, z up = absolute DTM elevation, origin at the SW corner of the
patch); the mesh is in Unity world coordinates (Y up). With the Terrain at Position (px, py, pz):
	x_unity = E + px,   y_unity = U,   z_unity = N + pz
(Terrain rotation/scale are ignored by Unity; the Terrain Y only places it, the heights are already absolute).
The GT is moved into the Unity frame, the meshes are left untouched.

The GT is restricted to the cells with real DTM data (valid_mask.npy) and to the area seen by the sonar up to
the time of each snapshot (cells within --coverage-radius of a CSV point, in plan): recall counts only
seabed that could have been reconstructed.

Output in the run folder: metrics_online.csv (one row per snapshot and tau) and gt_covered.ply (GT of the last
snapshot in the Unity frame, to check the alignment visually).

Usage (from the repository root):
	python evaluate_online.py output/online/<run>
	python evaluate_online.py output/online/<run> --tau 0.2 0.5 1.0 --final-only
"""

import argparse
import glob
import os
import re

import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy import ndimage

from evaluate import compute_metrics, surface_metrics
from utils.pipeline import load_config


def load_gt(gt_dir: str, offset_x: float, offset_z: float):
	"""GT mesh in the Unity frame without the cells that have no DTM data, and its grid geometry."""
	gt = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "gt_mesh.ply"))
	meta = yaml.safe_load(open(os.path.join(gt_dir, "meta.json")))
	valid = np.load(os.path.join(gt_dir, "valid_mask.npy")).reshape(-1)
	enu = np.asarray(gt.vertices)
	gt.vertices = o3d.utility.Vector3dVector(np.c_[enu[:, 0] + offset_x, enu[:, 2], enu[:, 1] + offset_z])
	triangles = np.asarray(gt.triangles)
	gt.remove_triangles_by_mask(~valid[triangles].all(axis=1))
	gt.remove_unreferenced_vertices()
	return gt, meta


def coverage_grid(x: np.ndarray, z: np.ndarray, origin_x: float, origin_z: float, cell: float, n: int, radius: float):
	"""Boolean n x n grid (row = z, column = x) of the cells within radius of a scan point, in plan."""
	cols = np.floor((x - origin_x) / cell).astype(int)
	rows = np.floor((z - origin_z) / cell).astype(int)
	inside = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
	seen = np.zeros((n, n), dtype=bool)
	seen[rows[inside], cols[inside]] = True
	r = int(np.ceil(radius / cell))
	if r > 0:
		yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
		seen = ndimage.binary_dilation(seen, structure=(xx ** 2 + yy ** 2) * cell ** 2 <= radius ** 2)
	return seen


def crop_to_seen(gt: o3d.geometry.TriangleMesh, seen: np.ndarray, origin_x: float, origin_z: float, cell: float):
	"""GT triangles whose centroid falls in a seen cell."""
	mesh = o3d.geometry.TriangleMesh(gt)
	centers = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)
	n = seen.shape[0]
	cols = np.clip(np.floor((centers[:, 0] - origin_x) / cell).astype(int), 0, n - 1)
	rows = np.clip(np.floor((centers[:, 2] - origin_z) / cell).astype(int), 0, n - 1)
	mesh.remove_triangles_by_mask(~seen[rows, cols])
	mesh.remove_unreferenced_vertices()
	return mesh


def main():
	config = load_config()
	parser = argparse.ArgumentParser(description="Metrics of an online reconstruction against the real-seabed GT")
	parser.add_argument("run", help="online_reconstruction.py run folder")
	parser.add_argument("--gt", default="data/real/patch_malamocco", help="folder with gt_mesh.ply, valid_mask.npy, meta.json")
	parser.add_argument("--terrain-position", type=float, nargs=2, default=[-128.0, -128.0], metavar=("X", "Z"),
						help="Position X and Z of the Unity Terrain (default -128 -128)")
	parser.add_argument("--csv", default=None, help="Unity CSV of the run (default: the one in <run>/config.txt)")
	parser.add_argument("--tau", type=float, nargs="+", default=[config["evaluation"]["tau"]])
	parser.add_argument("--n-samples", type=int, default=config["evaluation"]["n_samples"])
	parser.add_argument("--coverage-radius", type=float, default=1.0, help="m around the scan points counted as seen")
	parser.add_argument("--final-only", action="store_true", help="only mesh.ply, not every snapshot")
	args = parser.parse_args()

	px, pz = args.terrain_position
	gt, meta = load_gt(args.gt, px, pz)
	cell, n = meta["cell_m"], meta["samples"]
	origin_x, origin_z = px - cell / 2, pz - cell / 2  # GT vertices are cell centres -> cells centred on them

	csv_path = args.csv or yaml.safe_load(open(os.path.join(args.run, "config.txt")))["csv"]
	scan = pd.read_csv(csv_path, usecols=["x", "z", "elapsed_s"])
	updates = pd.read_csv(os.path.join(args.run, "updates.csv")).set_index("update")

	if args.final_only:
		meshes = [(int(updates.index.max()), os.path.join(args.run, "mesh.ply"))]
	else:
		paths = sorted(glob.glob(os.path.join(args.run, "meshes", "mesh_*.ply")))
		meshes = [(int(re.search(r"mesh_(\d+)\.ply$", p).group(1)), p) for p in paths] or \
				 [(int(updates.index.max()), os.path.join(args.run, "mesh.ply"))]
	print(f"GT {args.gt} in the Unity frame (Terrain X={px}, Z={pz}), {len(meshes)} meshes, tau {args.tau}")

	rows, seabed = [], None
	for k, path in meshes:
		elapsed = float(updates.loc[k, "elapsed_s"])
		seen_points = scan[scan["elapsed_s"] <= elapsed]
		seen = coverage_grid(seen_points["x"].to_numpy(), seen_points["z"].to_numpy(),
							 origin_x, origin_z, cell, n, args.coverage_radius)
		seabed = crop_to_seen(gt, seen, origin_x, origin_z, cell)
		mesh = o3d.io.read_triangle_mesh(path)
		if len(mesh.triangles) == 0 or len(seabed.triangles) == 0:
			print(f"[{k}] empty mesh or no seen GT: skipped")
			continue
		metrics, distances = compute_metrics(mesh, seabed, args.tau[0], args.n_samples)
		base = {"update": k, "elapsed_s": elapsed, "n_triangles": len(mesh.triangles),
				"gt_area_m2": seabed.get_surface_area(), "chamfer": metrics["chamfer"],
				"rec2gt_median": metrics["rec2gt_median"], "gt2rec_median": metrics["gt2rec_median"]}
		for tau in args.tau:
			m = surface_metrics(distances["rec2gt"], distances["gt2rec"], tau)
			rows.append({**base, "tau": tau, "precision": m["precision"], "recall": m["recall"], "fscore": m["fscore"]})
		m0 = rows[-len(args.tau)]
		print(f"[{k}] t={elapsed:.0f} s  GT seen {base['gt_area_m2']:.0f} m2  chamfer {base['chamfer']:.3f} m  "
			  f"median rec->GT {base['rec2gt_median']:.3f} m  P/R/F@{args.tau[0]} "
			  f"{m0['precision']:.3f}/{m0['recall']:.3f}/{m0['fscore']:.3f}")
		if base["rec2gt_median"] > 2.0:
			print("    WARNING: median distance > 2 m -> check --terrain-position (or a mirrored Terrain import)")

	if not rows:
		raise SystemExit("No mesh evaluated")
	out = os.path.join(args.run, "metrics_online.csv")
	pd.DataFrame(rows).to_csv(out, index=False, float_format="%.3f")
	o3d.io.write_triangle_mesh(os.path.join(args.run, "gt_covered.ply"), seabed)
	print(f"Saved {out} and gt_covered.ply")


if __name__ == "__main__":
	main()
