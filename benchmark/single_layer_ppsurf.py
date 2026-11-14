"""
"Single layer" post-processing of the PPSurf meshes: removes the double shell and measures how close PPSurf
gets to Poisson without it. On a regular XZ grid it takes the first hit of a vertical ray from above (top) and
from below (bottom) and their midpoint (middle); each variant becomes a grid mesh measured against the GT
seabed, next to the original PPSurf mesh and Poisson. Works on the meshes already saved by evaluate.py (no GPU).

Output: benchmark/benchmark_output/single_layer_ppsurf.csv and
        benchmark/benchmark_output/ppsurf_single_layer/<id>_{top,bottom,middle}.ply

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/single_layer_ppsurf.py
	python benchmark/single_layer_ppsurf.py --split val --limit 2 --step 0.25
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distances_to_mesh, surface_metrics, list_terrains, family_of, families_from_manifest, latest_run
from utils.mesh import top_surface
from utils.pipeline import load_config

OUTPUT_FILE = "benchmark/benchmark_output/single_layer_ppsurf.csv"
MESH_FOLDER = "benchmark/benchmark_output/ppsurf_single_layer"


def first_vertical_hit(mesh: o3d.geometry.TriangleMesh, X: np.ndarray, Z: np.ndarray,
					   y_start: float, direction: float) -> np.ndarray:
	"""Y of the first hit of vertical rays from y_start (direction -1 = downwards); +-inf where they miss."""
	scene = o3d.t.geometry.RaycastingScene()
	scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	n = X.size
	rays = np.stack([X.ravel(), np.full(n, y_start), Z.ravel(),
					 np.zeros(n), np.full(n, direction), np.zeros(n)], axis=1).astype(np.float32)
	t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
	return (y_start + direction * t).reshape(X.shape)


def mesh_from_heightmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> o3d.geometry.TriangleMesh:
	"""Triangulated grid; triangles with a missing vertex (non-finite Y) are dropped."""
	nr, nc = Y.shape
	idx = np.arange(nr * nc).reshape(nr, nc)
	a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
	c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
	tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])  # normals towards +Y
	valid = np.isfinite(Y.ravel())
	tri = tri[valid[tri].all(axis=1)]
	vertices = np.stack([X.ravel(), np.where(valid, Y.ravel(), 0.0), Z.ravel()], axis=1)
	mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(tri))
	mesh.remove_unreferenced_vertices()
	mesh.compute_vertex_normals()
	return mesh


def measure(mesh: o3d.geometry.TriangleMesh, seabed: o3d.geometry.TriangleMesh,
			seabed_points: np.ndarray, n_samples: int, tau: float) -> dict:
	mesh_points = np.asarray(mesh.sample_points_uniformly(n_samples).points)
	return surface_metrics(distances_to_mesh(mesh_points, seabed), distances_to_mesh(seabed_points, mesh), tau)


def main():
	config = load_config()
	ev = config["evaluation"]

	parser = argparse.ArgumentParser(description="Single layer from the PPSurf meshes (double shell) and metrics.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--run_dir", default=None,
						help="folder of an evaluate.py run without SW (default: the most recent of the split)")
	parser.add_argument("--limit", type=int, default=1, help="terrains per family (0 = all)")
	parser.add_argument("--step", type=float, default=0.25, help="XZ grid step, m")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	eval_mesh_folder = os.path.join(args.run_dir or latest_run(ev["output_dir"], args.split, False), "meshes")
	families = families_from_manifest(ev["synthetic_dir"])
	terrains = [t for t in list_terrains(ev["synthetic_dir"], args.split, families, args.limit)
				if os.path.exists(os.path.join(eval_mesh_folder, f"{t}_ppsurf.ply"))]
	if not terrains:
		raise SystemExit(f"No PPSurf mesh in {eval_mesh_folder}: run evaluate.py --methods ppsurf first")
	os.makedirs(MESH_FOLDER, exist_ok=True)
	rows = []

	for i, terrain_id in enumerate(terrains, 1):
		print(f"\n=== [{i}/{len(terrains)}] {terrain_id} ===")
		gt = o3d.io.read_triangle_mesh(os.path.join(ev["synthetic_dir"], args.split, terrain_id,
													f"{terrain_id}_ground_truth.obj"))
		seabed = top_surface(gt)
		seabed_points = np.asarray(seabed.sample_points_uniformly(ev["n_samples"]).points)
		ppsurf = o3d.io.read_triangle_mesh(os.path.join(eval_mesh_folder, f"{terrain_id}_ppsurf.ply"))

		# grid on the common footprint of PPSurf and GT (outside the GT there is no seabed to measure)
		v_pps, v_gt = np.asarray(ppsurf.vertices), np.asarray(gt.vertices)
		lo = np.maximum(v_pps.min(axis=0), v_gt.min(axis=0))
		hi = np.minimum(v_pps.max(axis=0), v_gt.max(axis=0))
		X, Z = np.meshgrid(np.arange(lo[0], hi[0], args.step), np.arange(lo[2], hi[2], args.step))
		y_above, y_below = v_pps[:, 1].max() + 10, v_pps[:, 1].min() - 10

		top = first_vertical_hit(ppsurf, X, Z, y_above, -1.0)
		bottom = first_vertical_hit(ppsurf, X, Z, y_below, 1.0)
		top[~np.isfinite(top)] = np.nan
		bottom[~np.isfinite(bottom)] = np.nan
		thickness = top - bottom
		print(f"  grid {X.shape[1]}x{X.shape[0]} (step {args.step} m): rays with a hit {np.isfinite(top).mean():.1%}, "
			  f"median thickness {np.nanmedian(thickness):.2f} m")

		variants = {"top": top, "bottom": bottom, "middle": (top + bottom) / 2}
		meshes_to_measure = {"ppsurf_original": ppsurf}
		for name, Y in variants.items():
			mesh = mesh_from_heightmap(X, Y, Z)
			o3d.io.write_triangle_mesh(os.path.join(MESH_FOLDER, f"{terrain_id}_{name}.ply"), mesh)
			meshes_to_measure[f"ppsurf_{name}"] = mesh
		poisson_path = os.path.join(eval_mesh_folder, f"{terrain_id}_poisson.ply")
		if os.path.exists(poisson_path):
			meshes_to_measure["poisson"] = o3d.io.read_triangle_mesh(poisson_path)

		for variant, mesh in meshes_to_measure.items():
			m = measure(mesh, seabed, seabed_points, ev["n_samples"], args.tau)
			rows.append({"id": terrain_id, "family": family_of(terrain_id, families),
						 "variant": variant, "median_thickness": float(np.nanmedian(thickness)), **m})
			print(f"  {variant:17s} chamfer={m['chamfer']:.3f}  P={m['precision']:.3f}  "
				  f"R={m['recall']:.3f}  F={m['fscore']:.3f}")

	df = pd.DataFrame(rows)
	df.to_csv(OUTPUT_FILE, index=False, float_format="%.3f")
	print(f"\nResults in {OUTPUT_FILE}, meshes in {MESH_FOLDER}/")

	order = ["ppsurf_original", "ppsurf_top", "ppsurf_bottom", "ppsurf_middle", "poisson"]
	mean = df.groupby("variant")[["chamfer", "precision", "recall", "fscore"]].mean()
	print(f"\n=== Mean over {df['id'].nunique()} terrains (tau={args.tau} m) ===")
	print(mean.reindex([v for v in order if v in mean.index]).round(3).to_string())
	print("\n=== Chamfer per terrain ===")
	table = df.pivot_table(index="id", columns="variant", values="chamfer")
	print(table[[v for v in order if v in table.columns]].round(3).to_string())


if __name__ == "__main__":
	main()
