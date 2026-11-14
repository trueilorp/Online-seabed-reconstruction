"""Reconstruction pipeline used by evaluate.py and benchmark/: config loading, common preprocessing,
the per-method runners (RUNNERS) and reconstruct(), on the full scan or with a sliding window."""

import copy
import os
import time

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from utils.mesh import (
	load_point_cloud,
	point_cloud_columns,
	filter_dataframe,
	dataframe_to_point_cloud,
	clean_point_cloud,
	estimate_normals,
)
from scripts.poisson import reconstruct_poisson, clean_mesh
from scripts.ppsurf import subsample_if_needed, reconstruct_ppsurf

CONFIG_PATH = "config/config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
	with open(path, "r") as file:
		return yaml.safe_load(file)


def filters_from_config(config: dict) -> dict:
	"""Optional filters for real sonar data (classes, altitude, excluded zone); unused on synthetic terrains."""
	pre = config["preprocessing"]
	zone_x, zone_z = pre.get("exclude_zone_x"), pre.get("exclude_zone_z")
	return dict(
		exclude_classes=pre.get("exclude_classes") or [],
		altitude_min=pre.get("altitude_min"),
		altitude_max=pre.get("altitude_max"),
		exclude_zone_x=tuple(zone_x) if zone_x else None,
		exclude_zone_z=tuple(zone_z) if zone_z else None,
	)


def voxel_from_config(config: dict):
	"""Downsampling voxel side in metres, None if downsampling is disabled."""
	ds = config["preprocessing"]["downsampling"]
	return ds["voxel_size"] if ds["enabled"] else None


def clean_from_config(pcd: o3d.geometry.PointCloud, config: dict) -> o3d.geometry.PointCloud:
	outlier = config["preprocessing"]["outlier_removal"]
	return clean_point_cloud(
		pcd,
		voxel_size=voxel_from_config(config),
		nb_neighbors=outlier["nb_neighbors"] if outlier["enabled"] else None,
		std_ratio=outlier["std_ratio"],
	)


def load_and_clean_point_cloud(config: dict) -> o3d.geometry.PointCloud:
	pcd = load_point_cloud(config["paths"]["input_csv"], **filters_from_config(config))
	return clean_from_config(pcd, config)


def run_name_from_config(config: dict) -> str:
	"""Run name for the files of the neural methods: the CSV name, or paths.run_name if set (windows)."""
	paths = config["paths"]
	return paths.get("run_name") or os.path.splitext(os.path.basename(paths["input_csv"]))[0]


def config_for_terrain(base_config: dict, scan_path: str, method: str) -> dict:
	config = copy.deepcopy(base_config)
	config["paths"]["input_csv"] = scan_path
	config["paths"].pop("run_name", None)
	config["algorithm"] = method
	return config


def run_poisson(pcd: o3d.geometry.PointCloud, config: dict):
	p = config["poisson"]
	pcd = estimate_normals(pcd)

	start = time.time()
	mesh = reconstruct_poisson(pcd, depth=p["depth"], remove_low_density=p["remove_low_density"])
	compute_time = time.time() - start

	mesh = clean_mesh(mesh, min_component_size=p["min_component_size"])
	mesh.vertex_colors = o3d.utility.Vector3dVector()  # Poisson interpolates the point colours: save without
	mesh.compute_vertex_normals()
	return mesh, pcd, compute_time, "poisson"


def run_ppsurf(pcd: o3d.geometry.PointCloud, config: dict):
	pps = config["ppsurf"]
	pcd = subsample_if_needed(pcd, pps["max_points"])
	mesh, compute_time = reconstruct_ppsurf(
		pcd, run_name_from_config(config),
		ppsurf_dir=os.path.join(config["paths"]["neural_methods"], "ppsurf"),
		conda_env=pps["conda_env"], work_dir=pps["work_dir"],
		gen_resolution_global=pps["gen_resolution_global"],
		rec_batch_size=pps["rec_batch_size"],
		timeout_s=pps.get("timeout_s"),
		gen_subsample_manifold_iter=pps.get("gen_subsample_manifold_iter"),
		gen_refine_iter=pps.get("gen_refine_iter"),
	)
	return mesh, pcd, compute_time, "ppsurf"


RUNNERS = {
	"poisson": run_poisson,
	"ppsurf": run_ppsurf,
}


MIN_WINDOW_POINTS = 100  # windows with fewer points after cleaning are skipped


def ping_column(df: pd.DataFrame) -> str:
	lower = {c.lower(): c for c in df.columns}
	col = lower.get("ping_number") or lower.get("ping number")
	if col is None:
		raise ValueError(f"Sliding window: no ping column in the CSV. Columns: {list(df.columns)}")
	return col


def ping_windows(csv_path: str, pings_per_window: int, overlap_pings: int, rows_per_chunk: int):
	"""
	Reads the CSV in chunks of rows (as they would arrive from the sonar) and yields windows of
	pings_per_window consecutive pings, overlapping by overlap_pings. Only the current window is
	kept in memory.

	Yields (window_df, own_pings, ping_col). own_pings are the pings whose mesh the window "owns":
	the overlap is split in half between consecutive windows, so each seabed area belongs to one window.
	"""
	if not 0 <= overlap_pings < pings_per_window:
		raise ValueError("Sliding window: requires 0 <= overlap_pings < pings_per_window")
	step = pings_per_window - overlap_pings
	half_left = overlap_pings // 2
	half_right = overlap_pings - half_left

	buffer, ping_col, first = None, None, True
	for chunk in pd.read_csv(csv_path, chunksize=rows_per_chunk):
		if ping_col is None:
			ping_col = ping_column(chunk)
		buffer = chunk if buffer is None else pd.concat([buffer, chunk], ignore_index=True)
		pings = pd.unique(buffer[ping_col])  # in arrival order
		# a window is complete only once a later ping has arrived
		while len(pings) > pings_per_window:
			window_pings = pings[:pings_per_window]
			start = 0 if first else half_left
			yield (buffer[buffer[ping_col].isin(window_pings)],
				   window_pings[start:pings_per_window - half_right], ping_col)
			buffer = buffer[~buffer[ping_col].isin(pings[:step])]
			pings = pings[step:]
			first = False

	if buffer is not None and len(buffer):  # last window: keeps everything left
		yield buffer, pings[0 if first else half_left:], ping_col


def _closer_to_own(query: np.ndarray, own: o3d.geometry.PointCloud,
				   others: o3d.geometry.PointCloud) -> np.ndarray:
	"""True for the query points closer to the window's own points than to the overlap points."""
	if len(others.points) == 0:
		return np.ones(len(query), dtype=bool)
	q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query))
	return np.asarray(q.compute_point_cloud_distance(own)) <= np.asarray(q.compute_point_cloud_distance(others))


def crop_far_from_points(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud,
						 max_distance) -> o3d.geometry.TriangleMesh:
	"""Removes the triangles whose centroid is farther than max_distance from the nearest point of pcd
	(surface invented far from the data). None = no crop."""
	if not max_distance or len(mesh.triangles) == 0:
		return mesh
	vertices = np.asarray(mesh.vertices)
	centroids = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[np.asarray(mesh.triangles)].mean(axis=1)))
	mesh.remove_triangles_by_mask(np.asarray(centroids.compute_point_cloud_distance(pcd)) > max_distance)
	mesh.remove_unreferenced_vertices()
	return mesh


def max_distance_from_config(config: dict):
	return config.get("postprocessing", {}).get("max_point_distance")


def _crop_to_own_area(mesh: o3d.geometry.TriangleMesh, own: o3d.geometry.PointCloud,
					  others: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
	"""Keeps only the triangles in the window's own area (avoids double surfaces)."""
	vertices = np.asarray(mesh.vertices)
	centroids = vertices[np.asarray(mesh.triangles)].mean(axis=1)
	mesh.remove_triangles_by_mask(~_closer_to_own(centroids, own, others))
	mesh.remove_unreferenced_vertices()
	return mesh


def reconstruct_in_windows(config: dict, window_stats: list = None):
	"""
	Sliding-window reconstruction: each window of pings is filtered, cleaned, reconstructed with the
	chosen method and cropped to its own area; the parts are concatenated. Only a small sample of the
	point cloud is kept per window.

	If window_stats is a list, one dict per window is appended to it with:
	  latency_s          from the moment the window is complete to when its mesh is merged
	                     (filters, cleaning, reconstruction with method start-up, crop)
	  method_time_s      reconstruction only, as measured by the runner
	  acquisition_s      window duration: from its first to its last ping (elapsed_s column)
	  budget_s           time between the completion of the previous window and of this one, i.e. the
	                     time available before the next one arrives (first window: acquisition_s)
	  rtf                real-time factor = latency_s / budget_s: < 1 keeps up with the sonar
	"""
	sw = config["sliding_window"]
	method = config["algorithm"]
	filters = filters_from_config(config)
	voxel = voxel_from_config(config)
	base_name = run_name_from_config(config)
	rng = np.random.default_rng(0)

	total_mesh = o3d.geometry.TriangleMesh()
	samples, total_time, method_name = [], 0.0, method
	prev_end = None
	windows = ping_windows(config["paths"]["input_csv"], sw["pings_per_window"],
						   sw["overlap_pings"], sw["rows_per_chunk"])
	for k, (df, own_pings, ping_col) in enumerate(windows, 1):
		start = time.perf_counter()
		print(f"\n--- Window {k}: pings {df[ping_col].iloc[0]}-{df[ping_col].iloc[-1]}, {len(df)} points ---")
		lower = {c.lower(): c for c in df.columns}
		col_t = lower.get("elapsed_s")
		t_start = float(df[col_t].min()) if col_t else None
		t_end = float(df[col_t].max()) if col_t else None
		stat = {"window": k, "first_ping": int(df[ping_col].iloc[0]), "last_ping": int(df[ping_col].iloc[-1]),
				"n_points": len(df), "n_clean_points": None, "method_time_s": None, "status": "ok"}

		def record(status: str = "ok"):
			if window_stats is None:
				return
			latency = time.perf_counter() - start
			acquisition = t_end - t_start if col_t else None
			budget = (t_end - (prev_end if prev_end is not None else t_start)) if col_t else None
			window_stats.append({**stat, "status": status, "latency_s": latency, "acquisition_s": acquisition,
								 "budget_s": budget, "rtf": latency / budget if budget else None})

		columns = point_cloud_columns(df)
		df = filter_dataframe(df, columns, **filters)
		if df.empty:
			print("  window empty after the filters: skipped")
			record("empty")
			prev_end = t_end
			continue
		pcd = dataframe_to_point_cloud(df, columns)
		own = df[ping_col].isin(own_pings).to_numpy()
		points = np.asarray(pcd.points)
		pcd_own = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[own]))
		pcd_others = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[~own]))
		if voxel:  # only to speed up the crop
			pcd_own, pcd_others = pcd_own.voxel_down_sample(voxel), pcd_others.voxel_down_sample(voxel)

		pcd = clean_from_config(pcd, config)
		stat["n_clean_points"] = len(pcd.points)
		if len(pcd.points) < MIN_WINDOW_POINTS:
			print(f"  only {len(pcd.points)} points after cleaning: window skipped")
			record("too_few_points")
			prev_end = t_end
			continue

		window_config = {**config, "paths": {**config["paths"], "run_name": f"{base_name}_w{k:03d}"}}
		mesh, used_pcd, compute_time, method_name = RUNNERS[method](pcd, window_config)
		total_time += compute_time
		stat["method_time_s"] = compute_time
		mesh = _crop_to_own_area(mesh, pcd_own, pcd_others)
		# distance to THIS window's points: Poisson closes each window with a dome in the water
		total_mesh += crop_far_from_points(mesh, pcd_own + pcd_others, max_distance_from_config(config))
		record()
		prev_end = t_end

		used = np.asarray(used_pcd.points)
		used = used[_closer_to_own(used, pcd_own, pcd_others)]
		n = min(len(used), sw["sample_points_per_window"])
		samples.append(used[rng.choice(len(used), n, replace=False)])

	if len(total_mesh.triangles) == 0:
		raise RuntimeError("Sliding window: no window produced a mesh")
	total_mesh.remove_duplicated_vertices()
	total_mesh.compute_vertex_normals()
	sample = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(samples)))
	print(f"\nSliding window: {len(total_mesh.vertices)} vertices, {len(sample.points)} points kept")
	return total_mesh, sample, total_time, method_name


def reconstruct(config: dict, window_stats: list = None):
	"""Full reconstruction as configured (full scan or sliding window). Returns (mesh, pcd, time, method_name).
	With the sliding window, window_stats (if a list) receives the latency and times of each window."""
	if config.get("sliding_window", {}).get("enabled"):
		return reconstruct_in_windows(config, window_stats)
	pcd = load_and_clean_point_cloud(config)
	mesh, used_pcd, compute_time, name = RUNNERS[config["algorithm"]](pcd, config)
	# same crop as the sliding window, so the two modes stay comparable
	mesh = crop_far_from_points(mesh, pcd, max_distance_from_config(config))
	return mesh, used_pcd, compute_time, name
