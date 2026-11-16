"""
Times Open3D Poisson alone on the first online batch of a recorded Unity CSV, with different thread counts,
using the same steps as online_reconstruction.py (0.5 m voxel centroids, outlier removal, normals, depth 9).
Used to tell a slow Poisson from CPU contention (e.g. Unity running on the same machine).

Usage (from the repository root):
	python benchmark/poisson_threads.py <scan.csv> --pings 5 --threads -1 12 8 6 4 2 1 --repeat 2
	OMP_WAIT_POLICY=passive python benchmark/poisson_threads.py <scan.csv>
"""
import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mesh import clean_point_cloud, estimate_normals
from utils.pipeline import load_config


def main():
	p = argparse.ArgumentParser(description="Poisson time vs number of threads on the first online batch")
	p.add_argument("csv")
	p.add_argument("--pings", type=int, default=5, help="pings in the batch (online.update_every_pings)")
	p.add_argument("--threads", type=int, nargs="+", default=[-1, 12, 8, 6, 4, 2, 1], help="-1 = all cores")
	p.add_argument("--repeat", type=int, default=2)
	a = p.parse_args()
	config = load_config()
	voxel = config["preprocessing"]["downsampling"]["voxel_size"]
	outlier = config["preprocessing"]["outlier_removal"]

	rows = []
	for chunk in pd.read_csv(a.csv, chunksize=200000):
		rows.append(chunk[chunk["ping_number"] < a.pings])
		if chunk["ping_number"].max() >= a.pings:
			break
	points = pd.concat(rows)[["x", "y", "z"]].to_numpy(dtype=np.float64)
	keys = np.floor(points / voxel).astype(np.int64)
	cells = pd.DataFrame(points, columns=["x", "y", "z"]).groupby([keys[:, 0], keys[:, 1], keys[:, 2]]).mean()
	pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cells.to_numpy()))
	with contextlib.redirect_stdout(io.StringIO()):
		pcd = clean_point_cloud(pcd, voxel_size=None, nb_neighbors=outlier["nb_neighbors"], std_ratio=outlier["std_ratio"])
		pcd = estimate_normals(pcd)
	print(f"{len(points)} points -> {len(pcd.points)} voxels after cleaning, depth {config['poisson']['depth']}, "
		  f"cores {os.cpu_count()}, OMP_WAIT_POLICY={os.environ.get('OMP_WAIT_POLICY', '(default)')}")

	for n in a.threads:
		times = []
		for _ in range(a.repeat):
			t0 = time.perf_counter()
			mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
				pcd, depth=config["poisson"]["depth"], n_threads=n)
			times.append(time.perf_counter() - t0)
		print(f"n_threads {n:>3}: " + "  ".join(f"{t:6.2f} s" for t in times) + f"   ({len(mesh.triangles)} triangles)",
			  flush=True)


if __name__ == "__main__":
	main()
