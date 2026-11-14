"""Renders an orbit around one mesh (set MESH_PATH below) as PNG frames and joins them into an mp4 with ffmpeg."""

import os
import subprocess

import open3d as o3d

MESH_PATH = "output/mesh/Terrain_unity/mesh_poisson.ply"
FRAME_FOLDER = "orbit_frames"
OUTPUT_VIDEO = "orbit_poisson.mp4"
N_FRAMES = 240
FPS = 30                      # 240 frames / 30 fps = 8 s
START_ELEVATION_DEG = 55.0    # initial downward tilt (0 = horizontal, 90 = from above)
HORIZONTAL_TURNS = 2          # full turns around the horizontal axis over the frames
VERTICAL_TURNS = 2            # full turns around the vertical axis, simultaneous
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WHITE_BACKGROUND = True


def render_orbit_frames():
	os.makedirs(FRAME_FOLDER, exist_ok=True)

	mesh = o3d.io.read_triangle_mesh(MESH_PATH)
	if len(mesh.vertices) == 0:
		print(f"[WARNING] '{MESH_PATH}' has no vertices, or was not found.")
		return False
	mesh.compute_vertex_normals()

	vis = o3d.visualization.Visualizer()
	vis.create_window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT, visible=True)
	vis.add_geometry(mesh)

	render_opt = vis.get_render_option()
	render_opt.mesh_show_back_face = True
	render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
	if WHITE_BACKGROUND:
		render_opt.background_color = [1, 1, 1]

	ctrl = vis.get_view_control()

	# rotate() works in pixel units, not degrees: empirical factor (~3-4 per degree with Open3D defaults)
	PIXELS_PER_DEGREE = 3.5
	ctrl.rotate(0.0, START_ELEVATION_DEG * PIXELS_PER_DEGREE)
	vis.poll_events()
	vis.update_renderer()

	horizontal_step = 360.0 * HORIZONTAL_TURNS * PIXELS_PER_DEGREE / N_FRAMES
	vertical_step = 360.0 * VERTICAL_TURNS * PIXELS_PER_DEGREE / N_FRAMES

	print(f"Rendering {N_FRAMES} frames...")
	for i in range(N_FRAMES):
		ctrl.rotate(horizontal_step, vertical_step)
		vis.poll_events()
		vis.update_renderer()
		vis.capture_screen_image(os.path.join(FRAME_FOLDER, f"frame_{i:04d}.png"), do_render=True)
		if (i + 1) % 20 == 0:
			print(f"  {i + 1}/{N_FRAMES} frames rendered")

	vis.destroy_window()
	print(f"Frames saved to: {FRAME_FOLDER}/")
	return True


def make_video():
	command = [
		"ffmpeg", "-y",
		"-framerate", str(FPS),
		"-i", os.path.join(FRAME_FOLDER, "frame_%04d.png"),
		"-c:v", "libx264",
		"-pix_fmt", "yuv420p",
		OUTPUT_VIDEO,
	]
	print("\nComposing video with ffmpeg...")
	try:
		subprocess.run(command, check=True)
		print(f"Video saved to: {OUTPUT_VIDEO}")
	except FileNotFoundError:
		print("[WARNING] ffmpeg not found in PATH. Install it and retry,")
		print(f"or join the frames in {FRAME_FOLDER}/ with another tool.")
	except subprocess.CalledProcessError as e:
		print(f"[ERROR] ffmpeg returned an error: {e}")


if __name__ == "__main__":
	if render_orbit_frames():
		make_video()
