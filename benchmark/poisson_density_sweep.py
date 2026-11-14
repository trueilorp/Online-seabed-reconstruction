"""
Sweep of the Poisson density-trim quantile (poisson.remove_low_density) on the synthetic terrains. Poisson runs
once per terrain (same preprocessing and normals as evaluate.py); the raw mesh is then trimmed at each quantile,
cleaned and measured against the GT seabed. Writes benchmark/benchmark_output/poisson_density_sweep.csv (one
row per terrain and quantile) and prints the mean F-score per quantile and the best quantile per terrain/family.

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/poisson_density_sweep.py
	python benchmark/poisson_density_sweep.py --split val --limit 1
	python benchmark/poisson_density_sweep.py --quantiles 0.02 0.05 0.1
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distances_to_mesh, surface_metrics, list_terrains, family_of, families_from_manifest
from scripts.poisson import raw_poisson, trim_low_density, clean_mesh
from utils.mesh import estimate_normals, top_surface
from utils.pipeline import load_config, config_for_terrain, load_and_clean_point_cloud

DEFAULT_QUANTILES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35]
OUTPUT_FILE = "benchmark/benchmark_output/poisson_density_sweep.csv"


def main():
	base_config = load_config()
	ev = base_config["evaluation"]
	p = base_config["poisson"]

	parser = argparse.ArgumentParser(description="Sweep of poisson.remove_low_density on the synthetic terrains.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--limit", type=int, default=ev["limit"], help="terrains per family (0 = all)")
	parser.add_argument("--quantiles", type=float, nargs="+", default=DEFAULT_QUANTILES)
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	families = families_from_manifest(ev["synthetic_dir"])
	terrains = list_terrains(ev["synthetic_dir"], args.split, families, args.limit)
	rows = []

	for i, terrain_id in enumerate(terrains, 1):
		folder = os.path.join(ev["synthetic_dir"], args.split, terrain_id)
		print(f"\n=== [{i}/{len(terrains)}] {terrain_id} ===")
		config = config_for_terrain(base_config, os.path.join(folder, f"{terrain_id}_scan.csv"), "poisson")

		t0 = time.time()
		pcd = estimate_normals(load_and_clean_point_cloud(config))
		raw_mesh, density = raw_poisson(pcd, depth=p["depth"])
		print(f"  raw Poisson in {time.time() - t0:.1f}s")

		seabed = top_surface(o3d.io.read_triangle_mesh(os.path.join(folder, f"{terrain_id}_ground_truth.obj")))
		seabed_points = np.asarray(seabed.sample_points_uniformly(ev["n_samples"]).points)

		for q in args.quantiles:
			mesh = clean_mesh(trim_low_density(raw_mesh, density, q), min_component_size=p["min_component_size"])
			if len(mesh.triangles) == 0:
				print(f"  q={q}: empty mesh after trimming")
				continue
			mesh_points = np.asarray(mesh.sample_points_uniformly(ev["n_samples"]).points)
			m = surface_metrics(distances_to_mesh(mesh_points, seabed), distances_to_mesh(seabed_points, mesh), args.tau)
			rows.append({"id": terrain_id, "family": family_of(terrain_id, families),
						 "quantile": q, "density_threshold": float(np.quantile(density, q)), **m})
			print(f"  q={q:<5} chamfer={m['chamfer']:.3f}  P={m['precision']:.3f}  "
				  f"R={m['recall']:.3f}  F={m['fscore']:.3f}")

	df = pd.DataFrame(rows)
	os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
	df.to_csv(OUTPUT_FILE, index=False, float_format="%.3f")
	print(f"\nResults per terrain and quantile in {OUTPUT_FILE}")

	pd.set_option("display.width", 200)
	print(f"\n=== Mean over {df['id'].nunique()} terrains, per quantile (current in config: {p['remove_low_density']}) ===")
	print(df.groupby("quantile")[["chamfer", "precision", "recall", "fscore"]].mean().round(3).to_string())

	print("\n=== F-score per family and quantile ===")
	print(df.pivot_table(index="family", columns="quantile", values="fscore").round(3).to_string())

	best = df.loc[df.groupby("id")["fscore"].idxmax(), ["id", "family", "quantile", "fscore"]]
	print("\n=== Best quantile per terrain (if it varies a lot, an adaptive threshold is better) ===")
	print(best.to_string(index=False))


if __name__ == "__main__":
	main()
