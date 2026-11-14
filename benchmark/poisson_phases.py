"""
Poisson timed phase by phase (normals, reconstruction, mesh cleanup) with the parameters of
utils/pipeline.py:run_poisson; each phase is appended to the timing CSV. Called by time_poisson.sh.
Outside the timer it also saves, next to the output mesh, a density diagnostic for choosing
poisson.remove_low_density: mesh_poisson_density.ply (mesh before trimming, coloured by density) and
density_histogram.png (per-vertex density with the current threshold).

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/poisson_phases.py <input.ply> <timing.csv> <dataset> <repetition> <output_mesh.ply>
"""

import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mesh import estimate_normals
from utils.pipeline import load_config
from scripts.poisson import raw_poisson, trim_low_density, clean_mesh


def save_density_diagnostic(mesh: o3d.geometry.TriangleMesh, density: np.ndarray,
							quantile: float, folder: str, title: str):
	norm = (density - density.min()) / (density.max() - density.min() + 1e-12)
	coloured = o3d.geometry.TriangleMesh(mesh)
	coloured.vertex_colors = o3d.utility.Vector3dVector(plt.get_cmap("plasma")(norm)[:, :3])
	coloured.compute_vertex_normals()
	o3d.io.write_triangle_mesh(os.path.join(folder, "mesh_poisson_density.ply"), coloured)

	threshold = np.quantile(density, quantile)
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.hist(density, bins=150, color="#4c72b0")
	ax.set_yscale("log")  # a small hump of spurious surface stays visible next to the seabed
	ax.axvline(threshold, color="#c44e52", linestyle="--", label=f"current threshold: quantile {quantile} = {threshold:.2f}")
	for q in (0.05, 0.1, 0.2):
		ax.axvline(np.quantile(density, q), color="grey", linewidth=0.8, alpha=0.6)
	ax.set_xlabel("Poisson density per vertex")
	ax.set_ylabel("n. vertices (log scale)")
	ax.set_title(f"{title} - grey lines: quantiles 0.05 / 0.1 / 0.2")
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(folder, "density_histogram.png"), dpi=120)
	plt.close(fig)

	quantiles = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9]
	print("  density per quantile: " + ", ".join(f"q{q}={np.quantile(density, q):.2f}" for q in quantiles))
	print(f"  density diagnostic saved in {folder}")


def main():
	ply_path, timing_file, dataset, repetition, output_mesh = sys.argv[1:6]
	p = load_config()["poisson"]
	pcd = o3d.io.read_point_cloud(ply_path)

	phases = []
	t0 = time.time()
	pcd = estimate_normals(pcd)
	phases.append(("normals", time.time() - t0))

	t0 = time.time()
	raw_mesh, density = raw_poisson(pcd, depth=p["depth"])
	raw_time = time.time() - t0
	save_density_diagnostic(raw_mesh, density, p["remove_low_density"],
							os.path.dirname(os.path.abspath(output_mesh)), dataset)
	t0 = time.time()
	mesh = trim_low_density(raw_mesh, density, p["remove_low_density"])
	phases.append(("reconstruction", raw_time + time.time() - t0))

	t0 = time.time()
	mesh = clean_mesh(mesh, min_component_size=p["min_component_size"])
	phases.append(("mesh_cleanup", time.time() - t0))
	o3d.io.write_triangle_mesh(output_mesh, mesh)

	# surface metrics left empty: time_poisson.sh adds them on the full_process row
	date = datetime.now().astimezone().isoformat(timespec="seconds")  # same format as "date -Iseconds"
	with open(timing_file, "a") as f:
		for phase, seconds in phases:
			print(f"  >> poisson / {phase}: {seconds:.2f}s")
			f.write(f"{date},poisson,{dataset},{len(pcd.points)},{repetition},{phase},{seconds:.2f},,,,,,,\n")


if __name__ == "__main__":
	main()
