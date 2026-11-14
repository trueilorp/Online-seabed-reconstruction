"""
Tiled PPSurf: the terrain is split into XZ squares, reconstructed separately (all in one PPSurf subprocess, dataset
mode of pps.py) and merged. The double-shell thickness is ~1% of the scene side, so small squares thin it; the
middle layer removes the rest. Each square gets its core plus a margin per side; the margins are discarded when
merging. Variants measured against the GT seabed: tile_original, tile_middle, full_original / full_middle (the
PPSurf mesh of the latest evaluate.py run, if present) and poisson (if present).

Output: benchmark/benchmark_output/tile_ppsurf.csv (appended) and benchmark/benchmark_output/ppsurf_tile/

Usage (from the repository root, "ricostruzione3d" environment; PPSurf parameters from the config):
	python benchmark/tile_ppsurf.py                              # split from config, 1 terrain per family, 25 m squares
	python benchmark/tile_ppsurf.py --id rocky_007 --side 25 --margin 2.5
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.single_layer_ppsurf import first_vertical_hit, mesh_from_heightmap, measure
from evaluate import list_terrains, family_of, families_from_manifest, latest_run
from scripts.ppsurf import find_conda_env_python
from utils.mesh import top_surface
from utils.pipeline import load_config, config_for_terrain, load_and_clean_point_cloud

OUTPUT_FILE = "benchmark/benchmark_output/tile_ppsurf.csv"
MESH_FOLDER = "benchmark/benchmark_output/ppsurf_tile"
WORK_DIR = "benchmark/benchmark_output/ppsurf_tile_tmp"


def box_xz(x0: float, x1: float, z0: float, z1: float) -> o3d.geometry.AxisAlignedBoundingBox:
	"""XZ box, (practically) unbounded in Y."""
	return o3d.geometry.AxisAlignedBoundingBox(np.array([x0, -1e4, z0]), np.array([x1, 1e4, z1]))


def layers(mesh: o3d.geometry.TriangleMesh, X: np.ndarray, Z: np.ndarray):
	"""Y of the first hit from above and from below at the (X, Z) nodes; NaN where the ray misses."""
	y = np.asarray(mesh.vertices)[:, 1]
	top = first_vertical_hit(mesh, X, Z, y.max() + 10, -1.0)
	bottom = first_vertical_hit(mesh, X, Z, y.min() - 10, 1.0)
	top[~np.isfinite(top)] = np.nan
	bottom[~np.isfinite(bottom)] = np.nan
	return top, bottom


def reconstruct_squares(clouds: dict, run_name: str, ppsurf_dir: str, pps: dict):
	"""All clouds {name: pcd} in a single PPSurf subprocess (dataset mode of pps.py).
	Returns ({name: mesh}, seconds); squares without an output mesh are missing from the dict."""
	python_ppsurf = find_conda_env_python(pps["conda_env"], ppsurf_dir)
	folder = os.path.abspath(os.path.join(WORK_DIR, run_name))
	shutil.rmtree(folder, ignore_errors=True)
	dataset_folder = os.path.join(folder, "dataset")
	points_folder = os.path.join(dataset_folder, "04_pts_vis")  # layout expected by pps.py
	output_folder = os.path.join(folder, "results")
	os.makedirs(points_folder)

	# in dataset mode pps.py expects normalised input and does not map the meshes back to metres: apply the
	# same normalisation it uses for a single file (padding_factor 0.05) and invert it on output
	normalisation = {}
	for name, pcd in clouds.items():
		p = np.asarray(pcd.points)
		center = (p.min(axis=0) + p.max(axis=0)) / 2
		scale = float((p.max(axis=0) - p.min(axis=0)).max() * 1.05)
		normalisation[name] = (center, scale)
		points_only = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((p - center) / scale))
		o3d.io.write_point_cloud(os.path.join(points_folder, f"{name}.xyz.ply"), points_only)
	testset = os.path.join(dataset_folder, "testset.txt")
	with open(testset, "w") as f:
		f.write("\n".join(clouds) + "\n")

	cmd = [python_ppsurf, "pps.py", "rec", testset, output_folder,
		   "--model.init_args.gen_resolution_global", str(pps["gen_resolution_global"]),
		   "--model.init_args.rec_batch_size", str(pps["rec_batch_size"])]
	env = os.environ.copy()
	env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # see scripts/ppsurf.py

	print(f"\n[PPSurf] {len(clouds)} squares in a single subprocess (environment '{pps['conda_env']}')...")
	start = time.time()
	result = subprocess.run(cmd, cwd=ppsurf_dir, env=env, capture_output=True, text=True)
	elapsed = time.time() - start
	if result.returncode != 0:
		print("--- last lines of stdout ---\n" + result.stdout[-3000:])
		print("--- last lines of stderr ---\n" + result.stderr[-3000:])
		raise RuntimeError(f"PPSurf (subprocess) exited with code {result.returncode}")
	print(f"[PPSurf] finished in {elapsed:.1f}s ({elapsed / len(clouds):.1f}s per square)")

	# dataset mode: <out>/<model name>/<dataset folder>/meshes/<name>.xyz.ply
	meshes = {}
	for name in clouds:
		candidates = glob.glob(os.path.join(output_folder, "*", "dataset", "meshes", f"{name}.xyz.ply"))
		if candidates:
			m = o3d.io.read_triangle_mesh(candidates[0])
			center, scale = normalisation[name]
			m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * scale + center)
			meshes[name] = m
		else:
			print(f"  WARNING: no output mesh for {name}")
	return meshes, elapsed


def main():
	base_config = load_config()
	ev = base_config["evaluation"]
	pps = base_config["ppsurf"]

	parser = argparse.ArgumentParser(description="Tiled PPSurf (XZ squares) + middle layer.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--run_dir", default=None,
						help="folder of an evaluate.py run without SW (default: the most recent of the split)")
	parser.add_argument("--limit", type=int, default=1, help="terrains per family (0 = all)")
	parser.add_argument("--id", nargs="+", default=None, help="specific terrains (ignores --limit)")
	parser.add_argument("--side", type=float, default=25.0, help="side of the square given to PPSurf, m")
	parser.add_argument("--margin", type=float, default=2.5, help="overlap per side, discarded when merging, m")
	parser.add_argument("--min-points", type=int, default=200, help="squares with fewer points in the core are skipped")
	parser.add_argument("--step", type=float, default=0.25, help="XZ grid step of the middle layer, m")
	parser.add_argument("--gen-res", type=int, default=pps["gen_resolution_global"],
						help="gen_resolution_global per square (default: config)")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()
	pps = {**pps, "gen_resolution_global": args.gen_res}

	core = args.side - 2 * args.margin
	if core <= 0:
		raise SystemExit("--side must be greater than 2 * --margin")

	families = families_from_manifest(ev["synthetic_dir"])
	terrains = args.id or list_terrains(ev["synthetic_dir"], args.split, families, args.limit)
	try:  # the evaluate.py meshes are optional: without them only the tiled variants are measured
		eval_mesh_folder = os.path.join(args.run_dir or latest_run(ev["output_dir"], args.split, False), "meshes")
	except FileNotFoundError:
		eval_mesh_folder = ""
	ppsurf_dir = os.path.join(base_config["paths"]["neural_methods"], "ppsurf")
	os.makedirs(MESH_FOLDER, exist_ok=True)
	run = datetime.datetime.now().isoformat(timespec="seconds")
	rows = []

	for i, terrain_id in enumerate(terrains, 1):
		folder = os.path.join(ev["synthetic_dir"], args.split, terrain_id)
		run_name = f"{terrain_id}_tile{args.side:g}"
		print(f"\n=== [{i}/{len(terrains)}] {terrain_id} ===")

		config = config_for_terrain(base_config, os.path.join(folder, f"{terrain_id}_scan.csv"), "ppsurf")
		pcd = load_and_clean_point_cloud(config)
		points = np.asarray(pcd.points)
		lo, hi = points.min(axis=0), points.max(axis=0)
		nx = max(1, int(np.ceil((hi[0] - lo[0]) / core)))
		nz = max(1, int(np.ceil((hi[2] - lo[2]) / core)))

		cores, clouds = {}, {}
		for ix in range(nx):
			for iz in range(nz):
				x0, z0 = lo[0] + ix * core, lo[2] + iz * core
				core_box = box_xz(x0, x0 + core, z0, z0 + core)
				if len(core_box.get_point_indices_within_bounding_box(pcd.points)) < args.min_points:
					continue
				name = f"{terrain_id}_x{ix:02d}_z{iz:02d}"
				cores[name] = (ix, iz, core_box)
				clouds[name] = pcd.crop(box_xz(x0 - args.margin, x0 + core + args.margin,
											   z0 - args.margin, z0 + core + args.margin))
		print(f"[squares] {len(clouds)} of {nx}x{nz} (side {args.side:g} m, core {core:g} m, "
			  f"marching-cubes cells ~{args.side / (pps['gen_resolution_global'] - 1):.2f} m)")

		square_meshes, elapsed = reconstruct_squares(clouds, run_name, ppsurf_dir, pps)

		# merge: original = meshes cropped to their cores; middle = one heightmap on the global grid
		X, Z = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, args.step), np.arange(lo[2], hi[2] + 1e-9, args.step))
		ix_node = np.minimum(((X - lo[0]) // core).astype(int), nx - 1)
		iz_node = np.minimum(((Z - lo[2]) // core).astype(int), nz - 1)
		top, bottom = np.full(X.shape, np.nan), np.full(X.shape, np.nan)
		tile_original = o3d.geometry.TriangleMesh()
		for name, mesh in square_meshes.items():
			ix, iz, core_box = cores[name]
			tile_original += mesh.crop(core_box)
			sel = (ix_node == ix) & (iz_node == iz)
			top[sel], bottom[sel] = layers(mesh, X[sel], Z[sel])
		tile_original.compute_vertex_normals()
		tile_middle = mesh_from_heightmap(X, (top + bottom) / 2, Z)
		o3d.io.write_triangle_mesh(os.path.join(MESH_FOLDER, f"{run_name}_original.ply"), tile_original)
		o3d.io.write_triangle_mesh(os.path.join(MESH_FOLDER, f"{run_name}_middle.ply"), tile_middle)
		tile_thickness = float(np.nanmedian(top - bottom))

		variants = {"tile_original": (tile_original, tile_thickness), "tile_middle": (tile_middle, tile_thickness)}
		full_path = os.path.join(eval_mesh_folder, f"{terrain_id}_ppsurf.ply")
		if os.path.exists(full_path):
			full = o3d.io.read_triangle_mesh(full_path)
			top_f, bottom_f = layers(full, X, Z)
			full_thickness = float(np.nanmedian(top_f - bottom_f))
			variants["full_original"] = (full, full_thickness)
			variants["full_middle"] = (mesh_from_heightmap(X, (top_f + bottom_f) / 2, Z), full_thickness)
		poisson_path = os.path.join(eval_mesh_folder, f"{terrain_id}_poisson.ply")
		if os.path.exists(poisson_path):
			variants["poisson"] = (o3d.io.read_triangle_mesh(poisson_path), float("nan"))

		seabed = top_surface(o3d.io.read_triangle_mesh(os.path.join(folder, f"{terrain_id}_ground_truth.obj")))
		seabed_points = np.asarray(seabed.sample_points_uniformly(ev["n_samples"]).points)
		for variant, (mesh, thickness) in variants.items():
			m = measure(mesh, seabed, seabed_points, ev["n_samples"], args.tau)
			rows.append({"run": run, "id": terrain_id, "family": family_of(terrain_id, families),
						 "variant": variant, "side": args.side, "margin": args.margin,
						 "gen_resolution_global": pps["gen_resolution_global"], "n_squares": len(square_meshes),
						 "ppsurf_time_s": elapsed, "median_thickness": thickness, **m})
			print(f"  {variant:19s} chamfer={m['chamfer']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
				  f"F={m['fscore']:.3f}  shell thickness={thickness:.2f} m")

	df = pd.DataFrame(rows)
	previous = pd.read_csv(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else pd.DataFrame()
	pd.concat([previous, df], ignore_index=True).to_csv(OUTPUT_FILE, index=False, float_format="%.3f")
	print(f"\nResults in {OUTPUT_FILE} (run {run}), meshes in {MESH_FOLDER}/")

	order = ["full_original", "full_middle", "tile_original", "tile_middle", "poisson"]
	mean = df.groupby("variant")[["chamfer", "precision", "recall", "fscore", "median_thickness"]].mean()
	print(f"\n=== Mean over {df['id'].nunique()} terrains (tau={args.tau} m) ===")
	print(mean.reindex([v for v in order if v in mean.index]).round(3).to_string())


if __name__ == "__main__":
	main()
