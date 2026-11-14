"""Point-cloud helpers shared by the pipeline: CSV loading and filtering, cleaning, normals, and the
seabed (top surface) of a closed ground-truth mesh."""

import numpy as np
import open3d as o3d
import pandas as pd


def load_point_cloud(csv_path: str, exclude_classes: list = None,
					 altitude_min: float = None, altitude_max: float = None,
					 exclude_zone_x: tuple = None, exclude_zone_z: tuple = None) -> o3d.geometry.PointCloud:
	df = pd.read_csv(csv_path)
	columns = point_cloud_columns(df)
	df = filter_dataframe(df, columns, exclude_classes=exclude_classes,
						  altitude_min=altitude_min, altitude_max=altitude_max,
						  exclude_zone_x=exclude_zone_x, exclude_zone_z=exclude_zone_z)
	pcd = dataframe_to_point_cloud(df, columns)
	print(f"[1/5] Loaded {len(pcd.points)} points from {csv_path} "
		  f"(X={columns['x']}, Y={columns['y']}, Z={columns['z']})")
	return pcd


def point_cloud_columns(df: pd.DataFrame) -> dict:
	"""Names of the x/y/z, intensity and classification columns in the supported CSV formats
	(synthetic, Unity, Omniscan, Surveyor)."""
	lower = {c.lower(): c for c in df.columns}

	col_x = lower.get("x") or lower.get("easting (local m)")
	col_z = lower.get("z") or lower.get("northing (local m)")
	col_y = lower.get("y") or lower.get("altitude (m)")
	col_intensity = lower.get("intensity") or lower.get("intensita") or lower.get("power (db)")
	col_classification = lower.get("classification")

	if not (col_x and col_y and col_z):
		raise ValueError(f"No x,y,z (or easting/northing/altitude) columns. CSV columns: {list(df.columns)}")
	return {"x": col_x, "y": col_y, "z": col_z,
			"intensity": col_intensity, "classification": col_classification}


def filter_dataframe(df: pd.DataFrame, columns: dict, exclude_classes: list = None,
					 altitude_min: float = None, altitude_max: float = None,
					 exclude_zone_x: tuple = None, exclude_zone_z: tuple = None) -> pd.DataFrame:
	"""Filters on the raw data: classes, altitude range, excluded rectangular zone."""
	col_x, col_y, col_z = columns["x"], columns["y"], columns["z"]
	col_classification = columns["classification"]

	if exclude_classes and col_classification:
		n_before = len(df)
		df = df[~df[col_classification].isin(exclude_classes)]
		print(f"[1/5] Removed {n_before - len(df)} rows with classification in {exclude_classes} "
			  f"-> {len(df)} rows left")

	if altitude_min is not None or altitude_max is not None:
		n_before = len(df)
		if altitude_min is not None:
			df = df[df[col_y] >= altitude_min]
		if altitude_max is not None:
			df = df[df[col_y] <= altitude_max]
		print(f"[1/5] Removed {n_before - len(df)} rows outside the altitude range "
			  f"[{altitude_min}, {altitude_max}] -> {len(df)} rows left")

	if exclude_zone_x is not None and exclude_zone_z is not None:
		n_before = len(df)
		inside = ((df[col_x] >= exclude_zone_x[0]) & (df[col_x] <= exclude_zone_x[1]) &
				  (df[col_z] >= exclude_zone_z[0]) & (df[col_z] <= exclude_zone_z[1]))
		df = df[~inside]
		print(f"[1/5] Removed {n_before - len(df)} rows in the zone X{exclude_zone_x} x Z{exclude_zone_z} "
			  f"-> {len(df)} rows left")
	return df


def dataframe_to_point_cloud(df: pd.DataFrame, columns: dict) -> o3d.geometry.PointCloud:
	"""Open3D point cloud from the x/y/z columns, coloured by intensity (or by height if missing)."""
	points = df[[columns["x"], columns["y"], columns["z"]]].to_numpy(dtype=np.float64)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)

	col_intensity = columns["intensity"]
	if col_intensity:
		values = df[col_intensity].to_numpy(dtype=np.float64)
		print(f"  Colouring by: {col_intensity}")
	else:
		# without colours Open3D renders black; height is a sensible fallback
		values = points[:, 1]
		print("  No intensity column: colouring by height (Y)")
	norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
	pcd.colors = o3d.utility.Vector3dVector(np.stack([norm, norm, 1 - norm], axis=1))
	return pcd


def clean_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size=None, nb_neighbors: int = 20,
					  std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
	"""voxel_size=None skips downsampling, nb_neighbors=None skips outlier removal."""
	if voxel_size is None:
		pcd_ds = pcd
		print(f"[2/5] Downsampling disabled: {len(pcd_ds.points)} points")
	else:
		pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
		print(f"[2/5] After downsampling (voxel={voxel_size}): {len(pcd_ds.points)} points")

	if nb_neighbors is None:
		print("[2/5] Outlier removal disabled")
		return pcd_ds

	pcd_clean, _ = pcd_ds.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
	print(f"[2/5] Removed {len(pcd_ds.points) - len(pcd_clean.points)} outliers -> {len(pcd_clean.points)} points left")
	return pcd_clean


def estimate_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
	"""Normals for Poisson: tangent plane on the 30 nearest neighbours, oriented upwards (+Y).
	A fixed radius found no neighbours after 0.5 m downsampling; for a seabed seen from above
	"outside" is always the water, more robust than MST propagation."""
	pcd.estimate_normals()
	pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 1.0, 0.0]))
	print("[3/5] Normals estimated and oriented")
	return pcd


def top_surface(gt_mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
	"""Seabed only of a closed GT (synthetic terrains): drops walls and bottom, i.e. faces touching the base level."""
	vertices = np.asarray(gt_mesh.vertices)
	triangles = np.asarray(gt_mesh.triangles)
	at_base = np.isclose(vertices[:, 1], vertices[:, 1].min(), atol=1e-4)
	keep = ~at_base[triangles].any(axis=1)
	seabed = o3d.geometry.TriangleMesh(gt_mesh)
	seabed.remove_triangles_by_mask(~keep)
	seabed.remove_unreferenced_vertices()
	return seabed
