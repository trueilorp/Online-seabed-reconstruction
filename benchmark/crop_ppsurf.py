"""
Grid-vs-model diagnosis for PPSurf: reconstructs a square crop of a terrain (same preprocessing as the full
run) so the 129 marching-cubes grid gets ~0.2 m cells instead of ~0.8 m, then measures on the same inner
square: the full-terrain PPSurf mesh, the crop mesh (both original and single-layer "middle") and Poisson.
Needs the meshes of an evaluate.py run and launches one PPSurf run (GPU).

Output: benchmark/benchmark_output/crop_ppsurf.csv and benchmark/benchmark_output/ppsurf_crop/

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/crop_ppsurf.py                              # ridges_012, 25 m crop in the centre
	python benchmark/crop_ppsurf.py --id rocky_012 --side 25 --center 10 -5
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.single_layer_ppsurf import first_vertical_hit, mesh_from_heightmap, measure
from scripts.ppsurf import reconstruct_ppsurf
from utils.mesh import top_surface
from evaluate import latest_run
from utils.pipeline import load_config, config_for_terrain, load_and_clean_point_cloud

OUTPUT_FILE = "benchmark/benchmark_output/crop_ppsurf.csv"
MESH_FOLDER = "benchmark/benchmark_output/ppsurf_crop"
WORK_DIR = "benchmark/benchmark_output/ppsurf_crop_tmp"


def square(center, side: float) -> o3d.geometry.AxisAlignedBoundingBox:
	"""XZ box centred in center, (practically) unbounded in Y."""
	cx, cz = center
	return o3d.geometry.AxisAlignedBoundingBox(np.array([cx - side / 2, -1e4, cz - side / 2]),
											   np.array([cx + side / 2, 1e4, cz + side / 2]))


def middle(mesh: o3d.geometry.TriangleMesh, box: o3d.geometry.AxisAlignedBoundingBox, step: float):
	"""Single layer (midpoint between first hit from above and from below) on a grid inside box."""
	lo, hi = box.get_min_bound(), box.get_max_bound()
	X, Z = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[2], hi[2] + 1e-9, step))
	y = np.asarray(mesh.vertices)[:, 1]
	top = first_vertical_hit(mesh, X, Z, y.max() + 10, -1.0)
	bottom = first_vertical_hit(mesh, X, Z, y.min() - 10, 1.0)
	top[~np.isfinite(top)] = np.nan
	bottom[~np.isfinite(bottom)] = np.nan
	return mesh_from_heightmap(X, (top + bottom) / 2, Z), float(np.nanmedian(top - bottom))


def main():
	base_config = load_config()
	ev = base_config["evaluation"]
	pps = base_config["ppsurf"]

	parser = argparse.ArgumentParser(description="PPSurf on a crop: is the 129 grid the limit?")
	parser.add_argument("--id", default="ridges_012")
	parser.add_argument("--split", default="val", choices=["train", "val", "test", "test+val"])
	parser.add_argument("--run_dir", default=None,
						help="folder of an evaluate.py run without SW (default: the most recent of the split)")
	parser.add_argument("--side", type=float, default=25.0, help="crop side, m")
	parser.add_argument("--margin", type=float, default=2.5, help="crop border excluded from the metrics, m")
	parser.add_argument("--center", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Z"))
	parser.add_argument("--step", type=float, default=0.25, help="grid step for the middle layer, m")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	folder = os.path.join(ev["synthetic_dir"], args.split, args.id)
	eval_mesh_folder = os.path.join(args.run_dir or latest_run(ev["output_dir"], args.split, False), "meshes")
	paths = {m: os.path.join(eval_mesh_folder, f"{args.id}_{m}.ply") for m in ("ppsurf", "poisson")}
	missing = [p for p in paths.values() if not os.path.exists(p)]
	if missing:
		raise SystemExit(f"Missing meshes (run evaluate.py first): {missing}")

	crop_box = square(args.center, args.side)
	inner_box = square(args.center, args.side - 2 * args.margin)
	run_name = f"{args.id}_crop{args.side:g}"

	config = config_for_terrain(base_config, os.path.join(folder, f"{args.id}_scan.csv"), "ppsurf")
	pcd = load_and_clean_point_cloud(config).crop(crop_box)
	print(f"\n[crop] {args.side:g}x{args.side:g} m around {args.center}: {len(pcd.points)} points "
		  f"(marching-cubes cells ~{args.side * 1.0 / (pps['gen_resolution_global'] - 1):.2f} m)")

	crop_mesh, elapsed = reconstruct_ppsurf(
		pcd, run_name,
		ppsurf_dir=os.path.join(base_config["paths"]["neural_methods"], "ppsurf"),
		conda_env=pps["conda_env"], work_dir=WORK_DIR,
		gen_resolution_global=pps["gen_resolution_global"], rec_batch_size=pps["rec_batch_size"],
	)

	seabed = top_surface(o3d.io.read_triangle_mesh(os.path.join(folder, f"{args.id}_ground_truth.obj")))
	seabed = seabed.crop(inner_box)
	seabed_points = np.asarray(seabed.sample_points_uniformly(ev["n_samples"]).points)

	full = o3d.io.read_triangle_mesh(paths["ppsurf"])
	full_middle, full_thickness = middle(full, inner_box, args.step)
	crop_middle, crop_thickness = middle(crop_mesh, inner_box, args.step)

	os.makedirs(MESH_FOLDER, exist_ok=True)
	o3d.io.write_triangle_mesh(os.path.join(MESH_FOLDER, f"{run_name}_original.ply"), crop_mesh)
	o3d.io.write_triangle_mesh(os.path.join(MESH_FOLDER, f"{run_name}_middle.ply"), crop_middle)

	variants = {
		"full_original": (full.crop(inner_box), full_thickness),
		"full_middle": (full_middle, full_thickness),
		"crop_original": (crop_mesh.crop(inner_box), crop_thickness),
		"crop_middle": (crop_middle, crop_thickness),
		"poisson": (o3d.io.read_triangle_mesh(paths["poisson"]).crop(inner_box), float("nan")),
	}
	rows = []
	print(f"\n=== {args.id}: metrics on the inner {args.side - 2 * args.margin:g} m square (tau={args.tau} m) ===")
	for name, (mesh, thickness) in variants.items():
		m = measure(mesh, seabed, seabed_points, ev["n_samples"], args.tau)
		rows.append({"id": args.id, "variant": name, "crop_side": args.side, "n_crop_points": len(pcd.points),
					 "median_thickness": thickness, "ppsurf_crop_time": elapsed, **m})
		print(f"  {name:19s} chamfer={m['chamfer']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
			  f"F={m['fscore']:.3f}  shell thickness={thickness:.2f} m")

	pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False, float_format="%.3f")
	print(f"\nPPSurf on the crop: {elapsed:.1f}s. Results in {OUTPUT_FILE}, meshes in {MESH_FOLDER}/")


if __name__ == "__main__":
	main()
