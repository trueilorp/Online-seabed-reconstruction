"""Legacy comparison of reconstructed meshes with the Unity ground truth: accuracy, completeness and coverage
table, a cross-section profile and distance histograms (set the paths below)."""

import os

import matplotlib
import numpy as np
import open3d as o3d
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = "Terrain_unity"
OUTPUT_DIR = f"output/comparison/{DATASET}"
GROUND_TRUTH_PATH = "data/sonar/Terrain_ground_truth.obj"
METHODS = {
	"poisson": f"output/mesh/{DATASET}/mesh_poisson.ply",
	"ppsurf": f"output/mesh/{DATASET}/mesh_ppsurf.ply",
}
N_ERROR_POINTS = 100_000      # points sampled for the histograms (on the GT and on each mesh)
N_SECTION_POINTS = 100        # resolution of the cross-section profile
Z_SECTION = None              # None = centre of the ground truth
COVERAGE_THRESHOLD = 0.5      # distance (m) below which a GT point counts as covered
OUTPUT_PREFIX = "comparison"


def make_raycasting_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
	scene = o3d.t.geometry.RaycastingScene()
	scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	return scene


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
	mesh = o3d.io.read_triangle_mesh(path)
	if len(mesh.vertices) == 0:
		raise ValueError(f"'{path}' has no vertices or was not found.")
	return mesh


# Completeness (GT -> reconstruction) penalises holes; accuracy (reconstruction -> GT) does not see them,
# so a method can trade one for the other: always report both.

def completeness(method_scene: o3d.t.geometry.RaycastingScene, gt_points: np.ndarray) -> np.ndarray:
	"""Distance of each ground-truth point from the reconstructed surface."""
	return method_scene.compute_distance(o3d.core.Tensor(gt_points, dtype=o3d.core.Dtype.Float32)).numpy()


def accuracy(method_mesh: o3d.geometry.TriangleMesh, gt_scene: o3d.t.geometry.RaycastingScene,
			 n_points: int) -> np.ndarray:
	"""Distance of each point sampled on the reconstructed mesh from the ground truth."""
	pcd = method_mesh.sample_points_uniformly(number_of_points=n_points)
	return gt_scene.compute_distance(o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)).numpy()


def coverage_percent(completeness_distances: np.ndarray, threshold: float) -> float:
	"""% of ground-truth points with a reconstruction within threshold metres."""
	return 100.0 * np.mean(completeness_distances <= threshold)


def compute_all_metrics(gt_scene, gt_points, method_meshes_scenes):
	"""method_meshes_scenes: name -> (mesh, scene). Returns (completeness per method, accuracy per method, table)."""
	completeness_per_method, accuracy_per_method, rows = {}, {}, []
	for name, (mesh, scene) in method_meshes_scenes.items():
		d_comp = completeness(scene, gt_points)
		d_acc = accuracy(mesh, gt_scene, N_ERROR_POINTS)
		completeness_per_method[name] = d_comp
		accuracy_per_method[name] = d_acc
		rows.append({
			"method": name,
			"accuracy_mean": d_acc.mean(),
			"accuracy_rms": np.sqrt(np.mean(d_acc ** 2)),
			"completeness_mean": d_comp.mean(),
			"completeness_rms": np.sqrt(np.mean(d_comp ** 2)),
			"completeness_hausdorff": d_comp.max(),
			f"coverage_pct_within_{COVERAGE_THRESHOLD}m": coverage_percent(d_comp, COVERAGE_THRESHOLD),
		})
	return completeness_per_method, accuracy_per_method, pd.DataFrame(rows)


def height_along_line(scene: o3d.t.geometry.RaycastingScene, x_values: np.ndarray,
					  z_fixed: float, y_start: float) -> np.ndarray:
	n = len(x_values)
	origins = np.stack([x_values, np.full(n, y_start), np.full(n, z_fixed)], axis=1)
	directions = np.tile([0.0, -1.0, 0.0], (n, 1))
	rays = o3d.core.Tensor(np.concatenate([origins, directions], axis=1), dtype=o3d.core.Dtype.Float32)
	heights = y_start - scene.cast_rays(rays)["t_hit"].numpy()
	heights[np.isinf(heights)] = np.nan  # no surface there
	return heights


def section_profile(gt_scene, method_scenes, gt_bbox):
	gt_min, gt_max = gt_bbox
	z_section = Z_SECTION if Z_SECTION is not None else (gt_min[2] + gt_max[2]) / 2
	x_values = np.linspace(gt_min[0], gt_max[0], N_SECTION_POINTS)
	y_start = gt_max[1] + 10  # above everything, to start outside the mesh

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(x_values, height_along_line(gt_scene, x_values, z_section, y_start), linestyle=":",
			color="lightgray", linewidth=2.5, label="Ground truth", zorder=10)
	for (name, scene), style in zip(method_scenes.items(), ["--", ":", "-.", "-"]):
		ax.plot(x_values, height_along_line(scene, x_values, z_section, y_start), style, linewidth=1.8, label=name)

	ax.set_xlabel("X")
	ax.set_ylabel("Height (Y)")
	ax.set_title(f"Cross-section at Z = {z_section:.2f}\n(gaps = no reconstructed surface at that point)")
	ax.legend()
	ax.grid(alpha=0.3)
	plt.tight_layout()
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	out = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_section_profile.png")
	plt.savefig(out, dpi=150)
	plt.close(fig)
	print(f"Cross-section saved to: {out}")


def histogram(errors_per_method: dict, title: str, file_name: str):
	fig, ax = plt.subplots(figsize=(9, 5))
	for name, distances in errors_per_method.items():
		ax.hist(distances, bins=80, alpha=0.55, label=f"{name} (mean {distances.mean():.3f} m)",
				range=(0, np.percentile(distances, 99)))
	ax.set_xlabel("Distance (m)")
	ax.set_ylabel("Point count")
	ax.set_title(title)
	ax.legend()
	plt.tight_layout()
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	out = os.path.join(OUTPUT_DIR, file_name)
	plt.savefig(out, dpi=150)
	plt.close(fig)
	print(f"Histogram saved to: {out}")


def main():
	print(f"Loading ground truth: {GROUND_TRUTH_PATH}")
	gt_mesh = load_mesh(GROUND_TRUTH_PATH)
	gt_scene = make_raycasting_scene(gt_mesh)
	gt_vertices = np.asarray(gt_mesh.vertices)
	gt_bbox = (gt_vertices.min(axis=0), gt_vertices.max(axis=0))
	gt_points = np.asarray(gt_mesh.sample_points_uniformly(number_of_points=N_ERROR_POINTS).points)

	method_meshes_scenes, method_scenes = {}, {}
	for name, path in METHODS.items():
		print(f"Loading {name}: {path}")
		mesh = load_mesh(path)
		scene = make_raycasting_scene(mesh)
		method_meshes_scenes[name] = (mesh, scene)
		method_scenes[name] = scene

	print("\nComputing accuracy, completeness and coverage for each method...")
	completeness_per_method, accuracy_per_method, _ = compute_all_metrics(gt_scene, gt_points, method_meshes_scenes)

	print("\nCross-section profile...")
	section_profile(gt_scene, method_scenes, gt_bbox)

	print("\nCompleteness histogram (GT -> reconstruction, penalises holes)...")
	histogram(completeness_per_method, "Completeness: distance from the ground truth to the nearest reconstruction",
			  f"{OUTPUT_PREFIX}_completeness_histogram.png")

	print("\nAccuracy histogram (reconstruction -> GT, blind to holes)...")
	histogram(accuracy_per_method, "Accuracy: distance from the reconstruction to the nearest ground truth",
			  f"{OUTPUT_PREFIX}_accuracy_histogram.png")

	print(f"\nDone. PNGs in {OUTPUT_DIR}/")


if __name__ == "__main__":
	main()
