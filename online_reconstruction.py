"""
Online Poisson reconstruction of the seabed from the CSV that Unity writes during the simulation
(x,y,z,intensity,ping_number,elapsed_s; Y up). The file is read while it grows; points are accumulated in a
voxel map and, every online.update_every_pings pings, the XZ box covered by the new points is rebuilt with the
Poisson pipeline, replaces that part of the mesh, and the mesh is rewritten.

Output in <online.output_dir>/<date_time>/: mesh.ply (latest merged mesh, rewritten at each update),
meshes/mesh_<update>.ply (a copy every --save-every updates, for visualization/online_gif.py), updates.csv
(one row per update: points, rebuilt box, latency, RTF), config.txt and log.txt.

Usage (from the repository root, "ricostruzione3d" environment):
	python online_reconstruction.py --csv /mnt/c/.../Recordings/scan_20261114_102506.csv   # Unity is writing it
	python online_reconstruction.py --latest /mnt/c/.../Recordings                       # waits for the new scan_*.csv
	python online_reconstruction.py --csv scan.csv --replay --speed 4                    # replay a recording
	python online_reconstruction.py --latest /mnt/c/.../Recordings --view                # live 3D window
"""

import os

os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import argparse
import contextlib
import copy
import datetime
import glob
import io
import shutil
import time

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from scripts.poisson import raw_poisson, trim_low_density, clean_mesh
from utils.mesh import point_cloud_columns, clean_point_cloud, estimate_normals
from utils.pipeline import (load_config, crop_far_from_points, max_distance_from_config, ping_column,
							MIN_WINDOW_POINTS)
from utils.run_info import TeeLog

PHASES = ["t_read", "t_parse", "t_voxel", "t_window", "t_filter", "t_normals", "t_poisson", "t_crop", "t_merge",
		  "t_write", "t_snapshot"]


def tail_csv(path: str, poll_s: float, idle_timeout_s: float, on_idle=None, times: dict = None):
	"""Yields DataFrames with the complete lines appended to a growing CSV; stops after idle_timeout_s
	without new data. on_idle() is called while waiting (e.g. to keep a window responsive).
	times["t_read"] / times["t_parse"] accumulate the time of the non-empty reads and of the parsing."""
	times = {} if times is None else times
	columns, rest, last_data = None, b"", time.time()
	with open(path, "rb") as f:
		while True:
			t0 = time.perf_counter()
			data = f.read()
			if data:
				times["t_read"] = times.get("t_read", 0.0) + time.perf_counter() - t0
			if not data:
				if time.time() - last_data > idle_timeout_s:
					return
				if on_idle:
					on_idle()
				time.sleep(poll_s)
				continue
			last_data = time.time()
			data = rest + data
			cut = data.rfind(b"\n") + 1  # the last line may still be half written
			lines, rest = data[:cut], data[cut:]
			if columns is None:
				if not lines:
					continue
				header, _, lines = lines.partition(b"\n")
				columns = header.decode().strip().split(",")
			if lines.strip():
				t0 = time.perf_counter()
				chunk = pd.read_csv(io.BytesIO(lines), header=None, names=columns)
				times["t_parse"] = times.get("t_parse", 0.0) + time.perf_counter() - t0
				yield chunk


def replay_csv(path: str, speed: float, rows_per_chunk: int = 5000, times: dict = None):
	"""Yields a finished recording in chunks at its original pace (elapsed_s), sped up by speed.
	times["t_parse"] accumulates the time of reading + parsing the chunks."""
	times = {} if times is None else times
	start = time.time()
	reader = iter(pd.read_csv(path, chunksize=rows_per_chunk))
	while True:
		t0 = time.perf_counter()
		chunk = next(reader, None)
		times["t_parse"] = times.get("t_parse", 0.0) + time.perf_counter() - t0
		if chunk is None:
			return
		wait = chunk["elapsed_s"].iloc[-1] / speed - (time.time() - start)
		if wait > 0:
			time.sleep(wait)
		yield chunk


def ping_batches(chunks, pings_per_batch: int, times: dict = None):
	"""Groups the incoming rows into batches of complete pings (a ping is complete once a later one arrives);
	the last, possibly partial, batch is yielded when the source ends. Its work is added to times["t_parse"]."""
	times = {} if times is None else times
	buffer, col = None, None
	for chunk in chunks:
		t0 = time.perf_counter()
		col = col or ping_column(chunk)
		buffer = chunk if buffer is None else pd.concat([buffer, chunk], ignore_index=True)
		pings = pd.unique(buffer[col])
		while len(pings) > pings_per_batch:
			done = pings[:pings_per_batch]
			batch = buffer[buffer[col].isin(done)]
			buffer = buffer[~buffer[col].isin(done)]
			pings = pings[pings_per_batch:]
			times["t_parse"] = times.get("t_parse", 0.0) + time.perf_counter() - t0
			yield batch
			t0 = time.perf_counter()
		times["t_parse"] = times.get("t_parse", 0.0) + time.perf_counter() - t0
	if buffer is not None and len(buffer):
		yield buffer


class VoxelMap:
	"""Running centroid of the points in each voxel: incremental downsampling, repeated passes are averaged."""

	def __init__(self, voxel: float):
		self.voxel = voxel
		self.cells = None

	def add(self, points: np.ndarray):
		keys = np.floor(points / self.voxel).astype(np.int64)
		batch = pd.DataFrame({"sx": points[:, 0], "sy": points[:, 1], "sz": points[:, 2], "n": 1.0},
							 index=pd.MultiIndex.from_arrays(keys.T)).groupby(level=[0, 1, 2]).sum()
		self.cells = batch if self.cells is None else self.cells.add(batch, fill_value=0.0)

	def centroids(self) -> np.ndarray:
		return self.cells[["sx", "sy", "sz"]].to_numpy() / self.cells[["n"]].to_numpy()


def in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
	"""Points whose X and Z fall inside the XZ box [lo, hi)."""
	x, z = points[:, 0], points[:, 2]
	return (x >= lo[0]) & (x < hi[0]) & (z >= lo[1]) & (z < hi[1])


def rebuild_region(lo: np.ndarray, hi: np.ndarray, centroids: np.ndarray, config: dict, times: dict = None):
	"""Poisson mesh of the XZ box [lo, hi) from the map voxels in the box plus a margin, cropped to the box by
	triangle centroid; None if there are too few points. Same steps as utils.pipeline.run_poisson, split so that
	each phase is timed into times (t_window, t_filter, t_normals, t_poisson, t_crop)."""
	times = {} if times is None else times
	p = config["poisson"]
	t0 = time.perf_counter()
	margin = config["online"]["margin"]
	points = centroids[in_box(centroids, lo - margin, hi + margin)]
	times["t_window"] = time.perf_counter() - t0
	if len(points) < MIN_WINDOW_POINTS:
		return None
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
	outlier = config["preprocessing"]["outlier_removal"]
	with contextlib.redirect_stdout(io.StringIO()):  # the per-step prints of the pipeline
		t0 = time.perf_counter()
		pcd = clean_point_cloud(pcd, voxel_size=None, std_ratio=outlier["std_ratio"],
								nb_neighbors=outlier["nb_neighbors"] if outlier["enabled"] else None)
		times["t_filter"] = time.perf_counter() - t0

		t0 = time.perf_counter()
		pcd = estimate_normals(pcd)
		times["t_normals"] = time.perf_counter() - t0

		t0 = time.perf_counter()
		mesh, density = raw_poisson(pcd, p["depth"], n_threads=config["online"]["poisson_threads"])
		times["t_poisson"] = time.perf_counter() - t0

		t0 = time.perf_counter()
		mesh = trim_low_density(mesh, density, p["remove_low_density"])
		mesh = clean_mesh(mesh, min_component_size=p["min_component_size"])
		mesh.vertex_colors = o3d.utility.Vector3dVector()
		mesh.compute_vertex_normals()
	mesh = crop_far_from_points(mesh, pcd, max_distance_from_config(config))
	mesh = remove_in_box(mesh, lo, hi, inside=False)
	times["t_crop"] = time.perf_counter() - t0
	return mesh


def remove_in_box(mesh: o3d.geometry.TriangleMesh, lo: np.ndarray, hi: np.ndarray, inside: bool = True):
	"""Removes the triangles whose centroid is inside (inside=True) or outside the XZ box."""
	if len(mesh.triangles) == 0:
		return mesh
	centers = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)].mean(axis=1)
	mask = in_box(centers, lo, hi)
	mesh.remove_triangles_by_mask(mask if inside else ~mask)
	mesh.remove_unreferenced_vertices()
	return mesh


def wait_for_csv(folder: str = None, path: str = None, poll_s: float = 0.5) -> str:
	"""Waits for the CSV to read: with folder, the newest *.csv modified after the launch (the recording Unity is
	writing now, not an old one); with path, that file to exist. Ctrl+C stops waiting."""
	start = time.time()
	print(f"Waiting for {'a new CSV in ' + folder if folder else path} (put the scene in Play; Ctrl+C to stop)...",
		  flush=True)
	while True:
		if path is not None:
			if os.path.exists(path):
				return path
		else:
			files = [f for f in glob.glob(os.path.join(folder, "*.csv")) if os.path.getmtime(f) >= start - 1.0]
			if files:
				return max(files, key=os.path.getmtime)
		time.sleep(poll_s)


class LiveView:
	def __init__(self):
		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window(window_name="Online reconstruction", width=1280, height=720, left=100, top=100)
		self.mesh = o3d.geometry.TriangleMesh()
		self.first = True
		self.vis.get_render_option().mesh_show_back_face = True

	def update(self, mesh: o3d.geometry.TriangleMesh):
		self.mesh.vertices, self.mesh.triangles = mesh.vertices, mesh.triangles
		self.mesh.compute_vertex_normals()
		self.mesh.paint_uniform_color([0.62, 0.70, 0.78])
		if self.first:  # added at the first update, so the camera frames the seabed
			self.vis.add_geometry(self.mesh)
			self.first = False
		else:
			self.vis.update_geometry(self.mesh)
		self.poll()

	def poll(self):
		self.vis.poll_events()
		self.vis.update_renderer()


def main():
	config = load_config()
	parser = argparse.ArgumentParser(description="Online Poisson reconstruction from the CSV written by Unity.")
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument("--csv", help="CSV written by Unity")
	source.add_argument("--latest", metavar="FOLDER", help="wait for a new CSV in FOLDER (modified after the launch) and read it")
	parser.add_argument("--replay", action="store_true", help="replay a finished CSV at its original pace")
	parser.add_argument("--speed", type=float, default=1.0, help="replay speed factor")
	parser.add_argument("--every", type=int, default=config["online"]["update_every_pings"],
						help="pings between two updates (default: online.update_every_pings)")
	parser.add_argument("--out", default=None, help="output folder (default: <online.output_dir>/<date_time>)")
	parser.add_argument("--view", action="store_true", help="show the mesh in a live Open3D window")
	parser.add_argument("--save-every", type=int, default=config["online"]["snapshot_every"],
						help="also keep a copy of the mesh every N updates in meshes/ (0 = never; for the GIF)")
	args = parser.parse_args()

	config = copy.deepcopy(config)
	config["online"]["update_every_pings"] = args.every
	config["online"]["snapshot_every"] = args.save_every
	on = config["online"]
	try:
		if args.replay:
			csv_path = args.csv or max(glob.glob(os.path.join(args.latest, "*.csv")), key=os.path.getmtime)
		else:
			csv_path = wait_for_csv(folder=args.latest, path=args.csv)
	except KeyboardInterrupt:
		raise SystemExit("\nStopped while waiting for the CSV")
	except ValueError:  # --replay --latest on a folder without CSV
		raise SystemExit(f"No CSV in {args.latest}")
	voxel = config["preprocessing"]["downsampling"]["voxel_size"]

	out = args.out or os.path.join(on["output_dir"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
	os.makedirs(out, exist_ok=True)
	tee = TeeLog(os.path.join(out, "log.txt")).__enter__()
	with open(os.path.join(out, "config.txt"), "w") as f:
		yaml.safe_dump({"csv": os.path.abspath(csv_path), "replay": args.replay, "speed": args.speed, **config},
					   f, sort_keys=False)
	print(f"Reading {csv_path} ({'replay x' + str(args.speed) if args.replay else 'live'}), output in {out}")

	view = LiveView() if args.view else None
	io_times = {}  # read/parse time accumulated since the previous batch
	chunks = (replay_csv(csv_path, args.speed, times=io_times) if args.replay
			  else tail_csv(csv_path, on["poll_s"], on["idle_timeout_s"], on_idle=view.poll if view else None,
							times=io_times))
	vmap = VoxelMap(voxel)
	merged, rows, prev_end, n_read = o3d.geometry.TriangleMesh(), [], None, 0
	mesh_path = os.path.join(out, "mesh.ply")
	if args.save_every:
		os.makedirs(os.path.join(out, "meshes"), exist_ok=True)

	try:
		for k, batch in enumerate(ping_batches(chunks, on["update_every_pings"], times=io_times), 1):
			arrived = time.time()
			start = time.perf_counter()
			times = {name: 0.0 for name in PHASES}
			times["t_read"], times["t_parse"] = io_times.pop("t_read", 0.0), io_times.pop("t_parse", 0.0)

			t0 = time.perf_counter()
			cols = point_cloud_columns(batch)
			points = batch[[cols["x"], cols["y"], cols["z"]]].to_numpy(dtype=np.float64)
			n_read += len(points)
			vmap.add(points)
			centroids = vmap.centroids()
			times["t_voxel"] = time.perf_counter() - t0

			lo, hi = points[:, [0, 2]].min(axis=0), points[:, [0, 2]].max(axis=0) + 1e-6
			region = rebuild_region(lo, hi, centroids, config, times)

			t0 = time.perf_counter()
			merged = remove_in_box(merged, lo, hi)
			if region is not None:
				merged += region
			merged.remove_duplicated_vertices()
			merged.compute_vertex_normals()
			times["t_merge"] = time.perf_counter() - t0

			t0 = time.perf_counter()
			tmp = os.path.join(out, "mesh.tmp.ply")
			o3d.io.write_triangle_mesh(tmp, merged)
			os.replace(tmp, mesh_path)  # readers never see a half-written mesh
			times["t_write"] = time.perf_counter() - t0
			ready = time.time()
			latency = time.perf_counter() - start

			t_start, t_end = float(batch["elapsed_s"].min()), float(batch["elapsed_s"].max())
			budget = t_end - (prev_end if prev_end is not None else t_start)
			prev_end = t_end
			ping_col = ping_column(batch)
			row = {"update": k, "first_ping": int(batch[ping_col].min()), "last_ping": int(batch[ping_col].max()),
				   "elapsed_s": t_end, "n_points_read": n_read, "n_voxels": len(centroids),
				   "box_m": f"{hi[0] - lo[0]:.0f}x{hi[1] - lo[1]:.0f}", "n_triangles": len(merged.triangles),
				   "latency_s": latency, "budget_s": budget, "rtf": latency / budget if budget > 0 else None}
			if args.save_every and k % args.save_every == 0:  # outside the timer
				t0 = time.perf_counter()
				shutil.copyfile(mesh_path, os.path.join(out, "meshes", f"mesh_{k:04d}.ply"))
				times["t_snapshot"] = time.perf_counter() - t0
			row.update(times)
			row.update({"arrived_unix_s": arrived, "ready_unix_s": ready})
			rows.append(row)
			pd.DataFrame(rows).to_csv(os.path.join(out, "updates.csv"), index=False, float_format="%.3f")
			print(f"[{k}] pings {row['first_ping']}-{row['last_ping']} (t={t_end:.1f} s): {len(centroids)} voxels, "
				  f"box {row['box_m']} m rebuilt, {len(merged.triangles)} triangles, latency {latency:.2f} s"
				  + (f", RTF {row['rtf']:.2f}" if row["rtf"] is not None else ""))
			if view:
				view.update(merged)
	except KeyboardInterrupt:
		print("\nStopped by the user")

	if rows:
		df = pd.DataFrame(rows)
		print(f"\n{len(df)} updates, {n_read} points read, median latency {df['latency_s'].median():.2f} s, "
			  f"max {df['latency_s'].max():.2f} s. Mesh: {mesh_path}")
	else:
		print("No complete ping received")
	tee.__exit__(None, None, None)
	if view:
		view.vis.run()  # keeps the last mesh on screen until the window is closed
		view.vis.destroy_window()


if __name__ == "__main__":
	main()
