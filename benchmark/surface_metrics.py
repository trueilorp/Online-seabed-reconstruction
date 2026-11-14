"""Chamfer, precision, recall and F-score of one mesh against the ground-truth seabed (same functions as
evaluate.py), printed as one CSV line "chamfer,precision,recall,fscore,tau" on stdout. Used by time_*.sh.

Usage (from the repository root, "ricostruzione3d" environment):
	python benchmark/surface_metrics.py <mesh> <gt.obj>
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distances_to_mesh, surface_metrics
from utils.mesh import top_surface
from utils.pipeline import load_config


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("mesh")
	parser.add_argument("gt")
	args = parser.parse_args()

	ev = load_config()["evaluation"]
	tau, n_samples = ev["tau"], ev["n_samples"]

	mesh = o3d.io.read_triangle_mesh(args.mesh)
	seabed = top_surface(o3d.io.read_triangle_mesh(args.gt))

	rec_points = np.asarray(mesh.sample_points_uniformly(n_samples).points)
	gt_points = np.asarray(seabed.sample_points_uniformly(n_samples).points)
	m = surface_metrics(distances_to_mesh(rec_points, seabed), distances_to_mesh(gt_points, mesh), tau)

	print(f"  chamfer={m['chamfer']:.4f} m, precision={m['precision']:.4f}, recall={m['recall']:.4f}, "
		  f"fscore={m['fscore']:.4f} (tau={tau} m)", file=sys.stderr)
	print(f"{m['chamfer']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['fscore']:.4f},{tau}")


if __name__ == "__main__":
	main()
