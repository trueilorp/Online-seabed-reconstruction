"""
Screenshot di tutte le mesh di una o piu' run di evaluate.py.

Per ogni cartella di run passata, apre ogni mesh in <run>/meshes/*.ply in una finestra
Open3D nascosta e salva l'immagine in <run>/screenshots/<nome mesh>.png.

La camera dipende solo dal terreno (dal bounding box del fondale della ground truth), non
dalla mesh: lo stesso terreno viene inquadrato allo stesso modo in tutte le run, quindi le
immagini di Poisson e PPSurf (con o senza SW) si possono confrontare direttamente.
Se la ground truth non si trova, si usa il bounding box della mesh.

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python visualization/screenshot_run.py output/evaluation/<run1> output/evaluation/<run2>
	python visualization/screenshot_run.py output/evaluation/<run> --vista alto --solo rocky_012 mixed_014
	python visualization/screenshot_run.py output/evaluation/<run> --ground_truth   # anche il fondale vero
	python visualization/screenshot_run.py --ground_truth --dati data/synthetic/test+val   # solo le .obj di una cartella

Con --ground_truth [CARTELLA] salva anche il fondale della ground truth di ogni terreno della run
(top_surface, senza pareti e fondo, stessa camera delle mesh) in CARTELLA/<id>_ground_truth.png
(default output/groundtruth). Ogni terreno viene salvato una volta sola, anche con piu' run.
Senza cartelle di run, con --ground_truth e --dati CARTELLA si salvano tutte le .obj trovate in
CARTELLA (anche nelle sottocartelle). Le mesh chiuse (GT sintetiche, con pareti e fondo) vengono
ridotte al fondale; quelle aperte (es. Terrain_ground_truth.obj di Unity) restano intere.
"""

import os

# su WSLg GLFW sceglie Wayland e la finestra (anche nascosta) non si crea -> forza X11
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

COLORE_MESH = [0.62, 0.70, 0.78]  # grigio-azzurro: il rilievo si legge dall'illuminazione
FOV_GRADI = 50.0
# direzione da cui guarda la camera (Y = verticale nei dati)
VISTE = {
	"obliqua": np.array([0.0, 1.0, 1.0]),  # 45 gradi dall'alto, lungo Z
	"alto": np.array([0.0, 1.0, 1e-3]),    # quasi zenitale (non esattamente, per avere un "su" definito)
}


def leggi_config_run(cartella_run: str) -> dict:
	"""La config salvata da evaluate.py in config.txt ({} se manca o non si legge)."""
	try:
		with open(os.path.join(cartella_run, "config.txt")) as f:
			return yaml.safe_load(f) or {}
	except (OSError, yaml.YAMLError):
		return {}


def fondale_terreno(config: dict, id_terreno: str, cache: dict):
	"""Fondale (top_surface) della GT del terreno, None se la GT non c'e'."""
	ev = config.get("evaluation", {})
	percorso = os.path.join(ev.get("synthetic_dir", "data/synthetic"), ev.get("split", ""),
							id_terreno, f"{id_terreno}_ground_truth.obj")
	if percorso not in cache:
		cache[percorso] = top_surface(o3d.io.read_triangle_mesh(percorso)) if os.path.exists(percorso) else None
	return cache[percorso]


def parametri_camera(bbox, larghezza: int, altezza: int, direzione: np.ndarray) -> o3d.camera.PinholeCameraParameters:
	"""Camera pinhole che guarda il centro del bbox da `direzione`, a distanza tale da inquadrarlo tutto."""
	centro = bbox.get_center()
	raggio = np.linalg.norm(bbox.get_extent()) / 2
	f = 0.5 * larghezza / np.tan(np.radians(FOV_GRADI) / 2)
	distanza = 1.0 * raggio / np.tan(np.radians(FOV_GRADI) / 2) * max(1.0, larghezza / altezza / 1.5)
	avanti = -direzione / np.linalg.norm(direzione)            # dalla camera verso il centro
	destra = np.cross(avanti, [0.0, 1.0, 0.0]); destra /= np.linalg.norm(destra)
	giu = np.cross(avanti, destra)                              # convenzione Open3D: y della camera verso il basso
	posizione = centro - avanti * distanza
	R = np.stack([destra, giu, avanti])
	estrinseca = np.eye(4)
	estrinseca[:3, :3] = R
	estrinseca[:3, 3] = -R @ posizione
	parametri = o3d.camera.PinholeCameraParameters()
	parametri.intrinsic = o3d.camera.PinholeCameraIntrinsic(larghezza, altezza, f, f,
															 larghezza / 2 - 0.5, altezza / 2 - 0.5)
	parametri.extrinsic = estrinseca
	return parametri


def screenshot(mesh: o3d.geometry.TriangleMesh, percorso_png: str, bbox, larghezza: int, altezza: int,
			   direzione) -> bool:
	if len(mesh.triangles) == 0:
		print(f"  [vuota] {percorso_png}")
		return False
	mesh = o3d.geometry.TriangleMesh(mesh)  # copia: il fondale in cache non va ricolorato
	mesh.compute_vertex_normals()
	mesh.paint_uniform_color(COLORE_MESH)

	vis = o3d.visualization.Visualizer()
	if not vis.create_window(width=larghezza, height=altezza, visible=False):
		raise RuntimeError("Finestra Open3D non creata (contesto OpenGL non disponibile)")
	vis.add_geometry(mesh)
	opzioni = vis.get_render_option()
	opzioni.mesh_show_back_face = True  # il guscio doppio di PPSurf si vede anche dal lato interno
	opzioni.background_color = np.array([1.0, 1.0, 1.0])
	vis.get_view_control().convert_from_pinhole_camera_parameters(
		parametri_camera(bbox or mesh.get_axis_aligned_bounding_box(), larghezza, altezza, direzione),
		allow_arbitrary=True)
	vis.poll_events()
	vis.update_renderer()
	vis.capture_screen_image(percorso_png, do_render=True)
	vis.destroy_window()
	return True


def salva_solo_ground_truth(args):
	"""Screenshot di tutte le .obj di una cartella (ricorsivo), senza passare da una run."""
	percorsi = sorted(glob.glob(os.path.join(args.dati, "**", "*.obj"), recursive=True))
	if not percorsi:
		sys.exit(f"Nessuna .obj in {args.dati}")
	suffisso_vista = "" if args.vista == "obliqua" else "_" + args.vista
	os.makedirs(args.ground_truth, exist_ok=True)
	print(f"{args.dati}: {len(percorsi)} file .obj")
	for percorso in percorsi:
		nome = os.path.splitext(os.path.basename(percorso))[0]
		if args.solo and not any(nome.startswith(i) for i in args.solo):
			continue
		mesh = o3d.io.read_triangle_mesh(percorso)
		# GT sintetiche chiuse: solo il fondale (pareti e fondo coprirebbero il terreno nella vista obliqua).
		# "Chiusa" = nessun bordo aperto (is_watertight controllerebbe anche le autointersezioni: lento)
		if len(mesh.get_non_manifold_edges(allow_boundary_edges=False)) == 0:
			mesh = top_surface(mesh)
		png = os.path.join(args.ground_truth, f"{nome}{suffisso_vista}.png")
		if screenshot(mesh, png, mesh.get_axis_aligned_bounding_box(),
					  args.larghezza, args.altezza, VISTE[args.vista]):
			print(f"  {png}")


def main():
	parser = argparse.ArgumentParser(description="Screenshot di tutte le mesh di una o piu' run di evaluate.py")
	parser.add_argument("run", nargs="*", help="cartelle di run (output/evaluation/<data_ora>_...)")
	parser.add_argument("--dati", default=None,
						help="senza cartelle di run: cartella di cui salvare tutte le .obj (es. data/synthetic/test+val)")
	parser.add_argument("--vista", choices=list(VISTE), default="obliqua")
	parser.add_argument("--solo", nargs="+", default=None, help="solo questi terreni (id)")
	parser.add_argument("--larghezza", type=int, default=1600)
	parser.add_argument("--altezza", type=int, default=1000)
	parser.add_argument("--ground_truth", nargs="?", const="output/groundtruth", default=None, metavar="CARTELLA",
						help="salva anche il fondale della GT di ogni terreno in CARTELLA (default output/groundtruth)")
	args = parser.parse_args()
	if not args.run:
		if not (args.ground_truth and args.dati):
			parser.error("indicare almeno una cartella di run, oppure --ground_truth con --dati")
		salva_solo_ground_truth(args)
		return

	cache_gt = {}
	suffisso_vista = "" if args.vista == "obliqua" else "_" + args.vista
	terreni_fatti = set()
	for cartella_run in args.run:
		mesh = sorted(glob.glob(os.path.join(cartella_run, "meshes", "*.ply")))
		if not mesh:
			print(f"Nessuna mesh in {cartella_run}/meshes: saltata")
			continue
		config = leggi_config_run(cartella_run)
		cartella_png = os.path.join(cartella_run, "screenshots")
		os.makedirs(cartella_png, exist_ok=True)
		print(f"{cartella_run}: {len(mesh)} mesh")
		for percorso in mesh:
			nome = os.path.splitext(os.path.basename(percorso))[0]  # <id>_<metodo>[_sw]
			id_terreno = nome.removesuffix("_sw").rsplit("_", 1)[0]  # i nomi dei metodi non hanno "_"
			if args.solo and id_terreno not in args.solo:
				continue
			fondale = fondale_terreno(config, id_terreno, cache_gt)
			bbox = fondale.get_axis_aligned_bounding_box() if fondale is not None else None
			nome_png = f"{nome}{suffisso_vista}.png"
			if screenshot(o3d.io.read_triangle_mesh(percorso), os.path.join(cartella_png, nome_png), bbox,
						  args.larghezza, args.altezza, VISTE[args.vista]):
				print(f"  {nome_png}")
			if args.ground_truth and fondale is not None and id_terreno not in terreni_fatti:
				terreni_fatti.add(id_terreno)
				nome_gt = f"{id_terreno}_ground_truth{suffisso_vista}.png"
				os.makedirs(args.ground_truth, exist_ok=True)
				if screenshot(fondale, os.path.join(args.ground_truth, nome_gt), bbox,
							  args.larghezza, args.altezza, VISTE[args.vista]):
					print(f"  {os.path.join(args.ground_truth, nome_gt)}")
		print(f"  -> {cartella_png}")


if __name__ == "__main__":
	main()
