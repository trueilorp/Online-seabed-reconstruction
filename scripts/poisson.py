"""Poisson surface reconstruction (Open3D) with low-density trimming and mesh cleanup."""

import numpy as np
import open3d as o3d


def reconstruct_poisson(pcd: o3d.geometry.PointCloud, depth: int = 9,
						remove_low_density: float = 0.02) -> o3d.geometry.TriangleMesh:
	"""depth: octree depth. remove_low_density: quantile of vertices with the lowest density to remove
	(Poisson invents surface where data are sparse)."""
	mesh, density = raw_poisson(pcd, depth)
	return trim_low_density(mesh, density, remove_low_density)


def raw_poisson(pcd: o3d.geometry.PointCloud, depth: int = 9):
	"""Poisson mesh before trimming, with the per-vertex density (low where the surface is invented)."""
	mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
	return mesh, np.asarray(density)


def trim_low_density(mesh: o3d.geometry.TriangleMesh, density: np.ndarray,
					 remove_low_density: float) -> o3d.geometry.TriangleMesh:
	"""Removes the vertices below the remove_low_density quantile of the density (on a copy)."""
	mesh = o3d.geometry.TriangleMesh(mesh)
	threshold = np.quantile(density, remove_low_density)
	mesh.remove_vertices_by_mask(density < threshold)
	print(f"[4/5] Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
	return mesh


def clean_mesh(mesh: o3d.geometry.TriangleMesh, min_component_size: int = 100) -> o3d.geometry.TriangleMesh:
	"""Removes degenerate triangles/vertices and small isolated components."""
	mesh.remove_degenerate_triangles()
	mesh.remove_duplicated_triangles()
	mesh.remove_duplicated_vertices()
	mesh.remove_non_manifold_edges()

	components, count_per_component, _ = mesh.cluster_connected_triangles()
	components = np.asarray(components)
	count_per_component = np.asarray(count_per_component)
	mesh.remove_triangles_by_mask(count_per_component[components] < min_component_size)
	mesh.remove_unreferenced_vertices()

	print(f"[5/5] Clean mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
	return mesh
