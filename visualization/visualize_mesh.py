"""Interactive Open3D viewer for one mesh (set MESH_PATH below); press S to save a screenshot."""

import datetime
import os

# on WSLg GLFW picks Wayland: the window does not appear or GLEW fails -> force X11
os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import numpy as np
import open3d as o3d

MESH_PATH = "..."
SHOW_WIREFRAME = False
SCREENSHOT_FOLDER = "output/screenshots_mesh"


def visualize(path: str, wireframe: bool = False):
	mesh = o3d.io.read_triangle_mesh(path)
	if len(mesh.vertices) == 0:
		print(f"[WARNING] '{path}' has no vertices, or was not found.")
		return
	mesh.compute_vertex_normals()

	print(f"Mesh loaded: {path}")
	print(f"  Vertices : {len(mesh.vertices)}")
	print(f"  Triangles: {len(mesh.triangles)}")
	print("\nControls: drag to rotate, wheel to zoom, right button to pan.")
	print("Press 'S' to save a screenshot of the current view. Close the window to exit.")

	os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

	vis = o3d.visualization.VisualizerWithKeyCallback()
	# explicit size/position: on WSLg the defaults (1920x1080) can end up off screen
	vis.create_window(window_name=path, width=1280, height=720, left=100, top=100)
	vis.add_geometry(mesh)

	if wireframe:
		lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
		lines.paint_uniform_color([0, 0, 0])
		vis.add_geometry(lines)

	render_opt = vis.get_render_option()
	if render_opt is None:
		print("[ERROR] OpenGL context not initialised (GLEW).")
		vis.destroy_window()
		return
	render_opt.mesh_show_back_face = True

	# oblique view from above: "up" is the axis with the smallest bbox extent (terrains are almost flat)
	up_axis = int(np.argmin(mesh.get_axis_aligned_bounding_box().get_extent()))
	up = np.zeros(3); up[up_axis] = 1.0
	horizontal = np.zeros(3); horizontal[(up_axis + 1) % 3] = 1.0
	ctr = vis.get_view_control()
	ctr.set_lookat(mesh.get_center())
	ctr.set_up(up)
	ctr.set_front(up + horizontal)
	ctr.set_zoom(0.6)
	render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

	def save_screenshot(vis):
		timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		base_name = os.path.splitext(os.path.basename(path))[0]
		output_path = os.path.join(SCREENSHOT_FOLDER, f"{base_name}_{timestamp}.png")
		vis.capture_screen_image(output_path, do_render=True)
		print(f"Screenshot saved to: {output_path}")
		return False

	vis.register_key_callback(ord("S"), save_screenshot)
	vis.run()
	vis.destroy_window()


if __name__ == "__main__":
	visualize(MESH_PATH, wireframe=SHOW_WIREFRAME)
