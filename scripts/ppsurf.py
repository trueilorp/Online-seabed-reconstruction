"""PPSurf reconstruction with the pretrained model, run as a subprocess in the "pps" conda environment
(the main pipeline runs in "ricostruzione3d")."""

import glob
import json
import os
import signal
import subprocess
import time

import open3d as o3d


def find_conda_env_python(env_name: str, ppsurf_dir: str) -> str:
	result = subprocess.run(["conda", "env", "list", "--json"], capture_output=True, text=True, check=True)
	for path in json.loads(result.stdout)["envs"]:
		if os.path.basename(path.rstrip("/")) == env_name:
			python_bin = os.path.join(path, "bin", "python")
			if not os.path.isfile(python_bin):
				raise FileNotFoundError(f"Python interpreter not found in {python_bin}")
			return python_bin
	raise RuntimeError(f"Conda environment '{env_name}' not found. Create it with: "
					   f"conda env create -f {ppsurf_dir}/pps.yml")


def subsample_if_needed(pcd: o3d.geometry.PointCloud, max_points: int) -> o3d.geometry.PointCloud:
	"""Safety net for very large clouds (threshold recommended by the PPSurf README)."""
	n = len(pcd.points)
	if n <= max_points:
		return pcd
	pcd_sub = pcd.random_down_sample(max_points / n)
	print(f"  PPSurf safety subsample: {n} -> {len(pcd_sub.points)} points (threshold {max_points})")
	return pcd_sub


def export_point_cloud(pcd: o3d.geometry.PointCloud, ply_path: str):
	"""Points only: PPSurf reads mesh.vertices and always ignores normals."""
	os.makedirs(os.path.dirname(ply_path), exist_ok=True)
	points_only = o3d.geometry.PointCloud()
	points_only.points = pcd.points
	o3d.io.write_point_cloud(ply_path, points_only)
	print(f"  Point cloud exported for PPSurf: {ply_path} ({len(pcd.points)} points)")


def reconstruct_ppsurf(pcd: o3d.geometry.PointCloud, run_name: str,
					   ppsurf_dir: str, conda_env: str, work_dir: str,
					   gen_resolution_global: int = 129, rec_batch_size: int = 10000,
					   timeout_s: float = None, gen_subsample_manifold_iter: int = None,
					   gen_refine_iter: int = None):
	"""Runs "pps.py rec" (pretrained 50NN model) and returns (mesh, compute_time_s).

	gen_subsample_manifold_iter / gen_refine_iter: test-time augmentation and marching-cubes refinement
	iterations; None = PPSurf default. timeout_s: the subprocess (with its workers) is killed after this
	time and an error is raised; None = no limit.
	"""
	python_ppsurf = find_conda_env_python(conda_env, ppsurf_dir)

	work_folder = os.path.abspath(os.path.join(work_dir, run_name))
	os.makedirs(work_folder, exist_ok=True)
	pc_path = os.path.join(work_folder, f"{run_name}.ply")
	output_folder = os.path.join(work_folder, "results")
	export_point_cloud(pcd, pc_path)

	cmd = [
		python_ppsurf, "pps.py", "rec", pc_path, output_folder,
		"--model.init_args.gen_resolution_global", str(gen_resolution_global),
		"--model.init_args.rec_batch_size", str(rec_batch_size),
	]
	if gen_subsample_manifold_iter is not None:
		cmd += ["--model.init_args.gen_subsample_manifold_iter", str(gen_subsample_manifold_iter)]
	if gen_refine_iter is not None:
		cmd += ["--model.init_args.gen_refine_iter", str(gen_refine_iter)]

	# must be set in the subprocess environment: without it marching cubes runs out of memory on a 4 GB GPU
	env = os.environ.copy()
	env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

	print(f"\n[PPSurf] Starting subprocess in environment '{conda_env}' (may take several minutes)...")
	print(f"  {' '.join(cmd)}")

	start = time.time()
	# own session: on timeout the PPSurf dataloader workers are killed too
	process = subprocess.Popen(cmd, cwd=ppsurf_dir, env=env, text=True,
							   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
	try:
		stdout, stderr = process.communicate(timeout=timeout_s)
	except subprocess.TimeoutExpired:
		os.killpg(process.pid, signal.SIGKILL)
		process.communicate()
		raise TimeoutError(f"PPSurf did not finish within {timeout_s:.0f} s: subprocess killed")
	result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
	compute_time = time.time() - start

	if result.returncode != 0:
		print("--- last lines of stdout ---")
		print(result.stdout[-3000:])
		print("--- last lines of stderr ---")
		print(result.stderr[-3000:])
		raise RuntimeError(f"PPSurf (subprocess) exited with code {result.returncode}")

	print(f"[PPSurf] Subprocess finished in {compute_time:.1f}s")

	# PPSurf writes {out_dir}/{basename(in_file)}/{basename(in_file)}.ply (undocumented)
	pattern = os.path.join(output_folder, os.path.basename(pc_path), "*.ply")
	candidates = glob.glob(pattern)
	if not candidates:
		raise FileNotFoundError(f"No .ply file found with pattern: {pattern}")

	mesh = o3d.io.read_triangle_mesh(candidates[0])
	mesh.compute_vertex_normals()
	print(f"[PPSurf] Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
	return mesh, compute_time
