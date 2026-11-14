"""
Off-screen screenshots of every mesh of one or more evaluate.py runs (<run>/meshes/*.ply ->
<run>/screenshots/<mesh name>.png), with a camera set by the terrain's ground-truth bounding box so the same
terrain is framed identically across runs. --ground_truth also saves the GT seabed of each terrain.

Usage (from the repository root, "ricostruzione3d" environment):
	python visualization/screenshot_run.py output/evaluation/<run1> output/evaluation/<run2>
	python visualization/screenshot_run.py output/evaluation/<run> --view top --only rocky_012 mixed_014
	python visualization/screenshot_run.py output/evaluation/<run> --ground_truth
	python visualization/screenshot_run.py --ground_truth --data data/synthetic/test+val
"""

import os

# on WSLg GLFW picks Wayland and the (hidden) window is not created -> force X11
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import argparse
import glob
import sys

import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mesh import top_surface

MESH_COLOR = [0.62, 0.70, 0.78]
FOV_DEG = 50.0
# direction the camera looks from (Y = vertical in the data)
VIEWS = {
	"oblique": np.array([0.0, 1.0, 1.0]),
	"top": np.array([0.0, 1.0, 1e-3]),  # not exactly zenithal, so that "up" is defined
}


def read_run_config(run: str) -> dict:
	"""Config saved by evaluate.py in config.txt ({} if missing or unreadable)."""
	try:
		with open(os.path.join(run, "config.txt")) as f:
			return yaml.safe_load(f) or {}
	except (OSError, yaml.YAMLError):
		return {}


def terrain_seabed(config: dict, terrain_id: str, cache: dict):
	"""Seabed (top_surface) of the terrain's GT, None if the GT is missing."""
	ev = config.get("evaluation", {})
	path = os.path.join(ev.get("synthetic_dir", "data/synthetic"), ev.get("split", ""),
						terrain_id, f"{terrain_id}_ground_truth.obj")
	if path not in cache:
		cache[path] = top_surface(o3d.io.read_triangle_mesh(path)) if os.path.exists(path) else None
	return cache[path]


def camera_parameters(bbox, width: int, height: int, direction: np.ndarray) -> o3d.camera.PinholeCameraParameters:
	"""Pinhole camera looking at the bbox centre from `direction`, far enough to frame all of it."""
	center = bbox.get_center()
	radius = np.linalg.norm(bbox.get_extent()) / 2
	f = 0.5 * width / np.tan(np.radians(FOV_DEG) / 2)
	distance = 1.0 * radius / np.tan(np.radians(FOV_DEG) / 2) * max(1.0, width / height / 1.5)
	forward = -direction / np.linalg.norm(direction)
	right = np.cross(forward, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
	down = np.cross(forward, right)  # Open3D convention: camera y points down
	position = center - forward * distance
	R = np.stack([right, down, forward])
	extrinsic = np.eye(4)
	extrinsic[:3, :3] = R
	extrinsic[:3, 3] = -R @ position
	params = o3d.camera.PinholeCameraParameters()
	params.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, width / 2 - 0.5, height / 2 - 0.5)
	params.extrinsic = extrinsic
	return params


def screenshot(mesh: o3d.geometry.TriangleMesh, png_path: str, bbox, width: int, height: int, direction) -> bool:
	if len(mesh.triangles) == 0:
		print(f"  [empty] {png_path}")
		return False
	mesh = o3d.geometry.TriangleMesh(mesh)  # copy: the cached seabed must not be recoloured
	mesh.compute_vertex_normals()
	mesh.paint_uniform_color(MESH_COLOR)

	vis = o3d.visualization.Visualizer()
	if not vis.create_window(width=width, height=height, visible=False):
		raise RuntimeError("Open3D window not created (no OpenGL context)")
	vis.add_geometry(mesh)
	options = vis.get_render_option()
	options.mesh_show_back_face = True  # the PPSurf double shell is visible from the inside too
	options.background_color = np.array([1.0, 1.0, 1.0])
	vis.get_view_control().convert_from_pinhole_camera_parameters(
		camera_parameters(bbox or mesh.get_axis_aligned_bounding_box(), width, height, direction),
		allow_arbitrary=True)
	vis.poll_events()
	vis.update_renderer()
	vis.capture_screen_image(png_path, do_render=True)
	vis.destroy_window()
	return True


def save_ground_truth_only(args):
	"""Screenshots of all .obj files of a folder (recursive), without a run."""
	paths = sorted(glob.glob(os.path.join(args.data, "**", "*.obj"), recursive=True))
	if not paths:
		sys.exit(f"No .obj in {args.data}")
	view_suffix = "" if args.view == "oblique" else "_" + args.view
	os.makedirs(args.ground_truth, exist_ok=True)
	print(f"{args.data}: {len(paths)} .obj files")
	for path in paths:
		name = os.path.splitext(os.path.basename(path))[0]
		if args.only and not any(name.startswith(i) for i in args.only):
			continue
		mesh = o3d.io.read_triangle_mesh(path)
		# closed synthetic GT (no open boundary): seabed only, walls and bottom would hide the terrain
		if len(mesh.get_non_manifold_edges(allow_boundary_edges=False)) == 0:
			mesh = top_surface(mesh)
		png = os.path.join(args.ground_truth, f"{name}{view_suffix}.png")
		if screenshot(mesh, png, mesh.get_axis_aligned_bounding_box(), args.width, args.height, VIEWS[args.view]):
			print(f"  {png}")


def main():
	parser = argparse.ArgumentParser(description="Screenshots of all meshes of one or more evaluate.py runs")
	parser.add_argument("run", nargs="*", help="run folders (output/evaluation/<date_time>_...)")
	parser.add_argument("--data", default=None,
						help="without run folders: folder whose .obj files to save (e.g. data/synthetic/test+val)")
	parser.add_argument("--view", choices=list(VIEWS), default="oblique")
	parser.add_argument("--only", nargs="+", default=None, help="only these terrains (ids)")
	parser.add_argument("--width", type=int, default=1600)
	parser.add_argument("--height", type=int, default=1000)
	parser.add_argument("--ground_truth", nargs="?", const="output/groundtruth", default=None, metavar="FOLDER",
						help="also save the GT seabed of each terrain in FOLDER (default output/groundtruth)")
	args = parser.parse_args()
	if not args.run:
		if not (args.ground_truth and args.data):
			parser.error("give at least one run folder, or --ground_truth with --data")
		save_ground_truth_only(args)
		return

	gt_cache = {}
	view_suffix = "" if args.view == "oblique" else "_" + args.view
	done_terrains = set()
	for run in args.run:
		meshes = sorted(glob.glob(os.path.join(run, "meshes", "*.ply")))
		if not meshes:
			print(f"No meshes in {run}/meshes: skipped")
			continue
		config = read_run_config(run)
		png_folder = os.path.join(run, "screenshots")
		os.makedirs(png_folder, exist_ok=True)
		print(f"{run}: {len(meshes)} meshes")
		for path in meshes:
			name = os.path.splitext(os.path.basename(path))[0]  # <id>_<method>[_sw]
			terrain_id = name.removesuffix("_sw").rsplit("_", 1)[0]  # method names have no "_"
			if args.only and terrain_id not in args.only:
				continue
			seabed = terrain_seabed(config, terrain_id, gt_cache)
			bbox = seabed.get_axis_aligned_bounding_box() if seabed is not None else None
			png_name = f"{name}{view_suffix}.png"
			if screenshot(o3d.io.read_triangle_mesh(path), os.path.join(png_folder, png_name), bbox,
						  args.width, args.height, VIEWS[args.view]):
				print(f"  {png_name}")
			if args.ground_truth and seabed is not None and terrain_id not in done_terrains:
				done_terrains.add(terrain_id)
				gt_name = f"{terrain_id}_ground_truth{view_suffix}.png"
				os.makedirs(args.ground_truth, exist_ok=True)
				if screenshot(seabed, os.path.join(args.ground_truth, gt_name), bbox,
							  args.width, args.height, VIEWS[args.view]):
					print(f"  {os.path.join(args.ground_truth, gt_name)}")
		print(f"  -> {png_folder}")


if __name__ == "__main__":
	main()
