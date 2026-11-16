"""
GIF of an online reconstruction (online_reconstruction.py): renders the mesh snapshots in <run>/meshes/ with a fixed
camera framed on the final mesh, writes the simulation time and the mesh size (from updates.csv) on each frame,
and joins them. Renders with Open3D (off-screen); without an OpenGL context it falls back to a matplotlib top
view coloured by depth.

Usage (from the repository root, "ricostruzione3d" environment):
	python visualization/online_gif.py output/online/<run>
	python visualization/online_gif.py output/online/<run> --view top --fps 3 --renderer matplotlib
"""

import os

# on WSLg GLFW picks Wayland and the (hidden) window is not created -> force X11
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import argparse
import glob
import io
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.screenshot_run import MESH_COLOR, VIEWS, camera_parameters


class Open3DRenderer:
	"""One hidden window reused for all frames, with the camera of the final mesh."""

	def __init__(self, final_mesh, width: int, height: int, view: str):
		self.vis = o3d.visualization.Visualizer()
		if not self.vis.create_window(width=width, height=height, visible=False):
			raise RuntimeError("no OpenGL context")
		options = self.vis.get_render_option()
		if options is None:
			raise RuntimeError("no OpenGL context")
		options.mesh_show_back_face = True
		options.background_color = np.array([1.0, 1.0, 1.0])
		self.camera = camera_parameters(final_mesh.get_axis_aligned_bounding_box(), width, height, VIEWS[view])

	def render(self, mesh) -> Image.Image:
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color(MESH_COLOR)
		self.vis.clear_geometries()
		self.vis.add_geometry(mesh)
		self.vis.get_view_control().convert_from_pinhole_camera_parameters(self.camera, allow_arbitrary=True)
		self.vis.poll_events()
		self.vis.update_renderer()
		buffer = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
		return Image.fromarray((buffer * 255).astype(np.uint8))

	def close(self):
		self.vis.destroy_window()


class MatplotlibRenderer:
	"""Top view coloured by depth, with the extent and colour range of the final mesh."""

	def __init__(self, final_mesh, width: int, height: int, view: str):
		v = np.asarray(final_mesh.vertices)
		self.xlim, self.zlim = (v[:, 0].min(), v[:, 0].max()), (v[:, 2].min(), v[:, 2].max())
		self.ylim = (v[:, 1].min(), v[:, 1].max())
		self.size = (width / 100, height / 100)

	def render(self, mesh) -> Image.Image:
		v, t = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
		fig, ax = plt.subplots(figsize=self.size, dpi=100)
		if len(t):
			tc = ax.tripcolor(v[:, 0], v[:, 2], t, v[:, 1], shading="gouraud", cmap="viridis",
							  vmin=self.ylim[0], vmax=self.ylim[1])
			fig.colorbar(tc, ax=ax, label="Y [m]", shrink=0.8)
		ax.set_xlim(*self.xlim)
		ax.set_ylim(*self.zlim)
		ax.set_aspect("equal")
		ax.set_xlabel("X [m]")
		ax.set_ylabel("Z [m]")
		fig.tight_layout()
		buffer = io.BytesIO()
		fig.savefig(buffer, format="png", facecolor="white")
		plt.close(fig)
		return Image.open(buffer).convert("RGB")

	def close(self):
		pass


def caption(image: Image.Image, text: str) -> Image.Image:
	draw = ImageDraw.Draw(image)
	try:
		font = ImageFont.load_default(size=max(16, image.height // 30))
	except TypeError:  # Pillow < 10.1
		font = ImageFont.load_default()
	draw.text((16, 12), text, fill=(31, 31, 30), font=font)
	return image


def main():
	parser = argparse.ArgumentParser(description="GIF of an online reconstruction from its mesh snapshots")
	parser.add_argument("run", help="folder of an online_reconstruction.py run (with meshes/ and updates.csv)")
	parser.add_argument("--out", default=None, help="output GIF (default: <run>/online.gif)")
	parser.add_argument("--view", choices=list(VIEWS), default="oblique")
	parser.add_argument("--renderer", choices=["auto", "open3d", "matplotlib"], default="auto")
	parser.add_argument("--fps", type=float, default=2.0)
	parser.add_argument("--hold", type=float, default=2.0, help="seconds the last frame stays on screen")
	parser.add_argument("--width", type=int, default=1200)
	parser.add_argument("--height", type=int, default=750)
	args = parser.parse_args()

	paths = sorted(glob.glob(os.path.join(args.run, "meshes", "mesh_*.ply")))
	if not paths:
		sys.exit(f"No snapshots in {args.run}/meshes: run online_reconstruction.py with --save-every 1")
	updates = pd.read_csv(os.path.join(args.run, "updates.csv")).set_index("update")
	final = o3d.io.read_triangle_mesh(paths[-1])

	renderer = None
	if args.renderer in ("auto", "open3d"):
		try:
			renderer = Open3DRenderer(final, args.width, args.height, args.view)
		except RuntimeError:
			if args.renderer == "open3d":
				raise
			print("Open3D could not open an OpenGL context: matplotlib top view instead")
	if renderer is None:
		renderer = MatplotlibRenderer(final, args.width, args.height, args.view)

	frames = []
	for path in paths:
		k = int(re.search(r"mesh_(\d+)\.ply$", path).group(1))
		mesh = o3d.io.read_triangle_mesh(path)
		frame = renderer.render(mesh)
		if k in updates.index:
			u = updates.loc[k]
			frame = caption(frame, f"t = {u['elapsed_s']:.0f} s   update {k}   pings {u['first_ping']}-{u['last_ping']}   "
								   f"{int(u['n_triangles']):,} triangles   latency {u['latency_s']:.1f} s")
		frames.append(frame)
		print(f"  frame {len(frames)}/{len(paths)}")
	renderer.close()

	out = args.out or os.path.join(args.run, "online.gif")
	duration = int(1000 / args.fps)
	durations = [duration] * (len(frames) - 1) + [duration + int(args.hold * 1000)]
	frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
	print(f"Saved {out} ({len(frames)} frames)")


if __name__ == "__main__":
	main()
