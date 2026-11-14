"""Alignment check between a Unity point-cloud CSV and the Terrain ground-truth .obj: bounding boxes, horizontal
and vertical overlap, and a top-view plot (set the paths below)."""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd

CSV_PATH = "data/sonar/Terrain_unity.csv"
OBJ_PATH = "data/sonar/Terrain_ground_truth.obj"
OUTPUT_PNG = "output/unity_ground_truth_comparison.png"


def load_csv_point_cloud(path: str):
	df = pd.read_csv(path)
	lower = {c.lower(): c for c in df.columns}
	col_x, col_y, col_z = lower.get("x"), lower.get("y"), lower.get("z")
	if not (col_x and col_y and col_z):
		raise ValueError(f"No x,y,z columns in the CSV. Columns: {list(df.columns)}")
	return df[[col_x, col_y, col_z]].to_numpy(dtype=np.float64), df


def print_bbox(name: str, min_xyz, max_xyz):
	print(f"\n{name}:")
	for i, axis in enumerate("XYZ"):
		print(f"  {axis}: [{min_xyz[i]:.3f}, {max_xyz[i]:.3f}]  (extent: {max_xyz[i] - min_xyz[i]:.3f})")


def main():
	print("=" * 60)
	print("Point cloud (Unity CSV)")
	print("=" * 60)
	points, _ = load_csv_point_cloud(CSV_PATH)
	print(f"Number of points: {len(points)}")
	if len(points) < 5000:
		print("[WARNING] Few points for a sensible Poisson reconstruction (usually thousands to tens of "
			  "thousands are needed). Consider recording more scans in Unity.")

	min_pc, max_pc = points.min(axis=0), points.max(axis=0)
	print_bbox("Point cloud bounding box", min_pc, max_pc)

	print("\n" + "=" * 60)
	print("Ground truth (Terrain .obj)")
	print("=" * 60)
	mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
	print(f"Vertices: {len(mesh.vertices)}")
	print(f"Triangles: {len(mesh.triangles)}")
	if len(mesh.vertices) == 0:
		print("[WARNING] The .obj has no vertices: check the path or the Unity export.")
		return

	vertices = np.asarray(mesh.vertices)
	min_gt, max_gt = vertices.min(axis=0), vertices.max(axis=0)
	print_bbox("Ground truth bounding box", min_gt, max_gt)

	print("\n" + "=" * 60)
	print("Alignment and coverage check")
	print("=" * 60)

	overlap_min_x, overlap_max_x = max(min_pc[0], min_gt[0]), min(max_pc[0], max_gt[0])
	overlap_min_z, overlap_max_z = max(min_pc[2], min_gt[2]), min(max_pc[2], max_gt[2])
	if overlap_max_x <= overlap_min_x or overlap_max_z <= overlap_min_z:
		print("[WARNING] NO overlap between the point cloud and the ground truth on the horizontal plane (X,Z): "
			  "likely a coordinate misalignment (different reference frames or a wrong transform).")
	else:
		area_pc = (max_pc[0] - min_pc[0]) * (max_pc[2] - min_pc[2])
		area_gt = (max_gt[0] - min_gt[0]) * (max_gt[2] - min_gt[2])
		area_overlap = (overlap_max_x - overlap_min_x) * (overlap_max_z - overlap_min_z)
		pct_gt_covered = 100 * area_overlap / area_gt if area_gt > 0 else 0
		pct_pc_inside = 100 * area_overlap / area_pc if area_pc > 0 else 0
		print(f"Point cloud bounding-box area (X-Z): {area_pc:.2f}")
		print(f"Ground truth bounding-box area (X-Z): {area_gt:.2f}")
		print(f"Overlap area: {area_overlap:.2f}")
		print(f"  -> {pct_gt_covered:.1f}% of the Terrain area is covered by the point cloud")
		print(f"  -> {pct_pc_inside:.1f}% of the point cloud falls inside the Terrain area")
		if pct_pc_inside < 50:
			print("[WARNING] Less than half of the point cloud falls inside the Terrain area: check the "
				  "coordinate alignment before trusting the comparison with the ground truth.")

	if min_pc[1] > max_gt[1] or max_pc[1] < min_gt[1]:
		print("[WARNING] The height range (Y) of the point cloud does not overlap the Terrain at all: "
			  "suspected vertical misalignment (e.g. sensor height not removed, or a different origin).")

	fig, ax = plt.subplots(figsize=(8, 8))
	ax.scatter(points[:, 0], points[:, 2], s=2, alpha=0.5, label="Point cloud (Unity)", color="steelblue")
	ax.plot([min_gt[0], max_gt[0], max_gt[0], min_gt[0], min_gt[0]],
			[min_gt[2], min_gt[2], max_gt[2], max_gt[2], min_gt[2]],
			color="darkorange", linewidth=2, label="Terrain extent (ground truth)")
	ax.set_xlabel("X")
	ax.set_ylabel("Z")
	ax.set_title("Top view: point cloud vs Terrain area")
	ax.set_aspect("equal", adjustable="box")
	ax.legend()
	plt.tight_layout()
	plt.savefig(OUTPUT_PNG, dpi=150)
	print(f"\nPlot saved to: {OUTPUT_PNG}")


if __name__ == "__main__":
	main()
