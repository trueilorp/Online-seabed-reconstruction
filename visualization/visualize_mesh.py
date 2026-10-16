import open3d as o3d

# --- Imposta qui il file da aprire ---
PERCORSO_MESH = "output/evaluation/val/meshes/flat_sand_012_ppsurf.ply" # "output/mesh/Terrain_unity/mesh_ppsurf.ply"
MOSTRA_WIREFRAME = False   # True per vedere anche gli spigoli dei triangoli sovrapposti
CARTELLA_SCREENSHOT = "output/screenshot_mesh"   # cartella dove salvare le immagini (creata se non esiste)


def visualizza(percorso: str, wireframe: bool = False):
	import os
	import datetime

	mesh = o3d.io.read_triangle_mesh(percorso)

	if len(mesh.vertices) == 0:
		print(f"[ATTENZIONE] '{percorso}' non contiene vertici, o non è stato trovato.")
		return

	mesh.compute_vertex_normals()

	print(f"Mesh caricata: {percorso}")
	print(f"  Vertici : {len(mesh.vertices)}")
	print(f"  Triangoli: {len(mesh.triangles)}")
	print(f"\nControlli: trascina per ruotare, rotella per zoomare, tasto destro per pan.")
	print(f"Premi 'S' per salvare uno screenshot della vista corrente. Chiudi la finestra per uscire.")

	os.makedirs(CARTELLA_SCREENSHOT, exist_ok=True)

	vis = o3d.visualization.VisualizerWithKeyCallback()
	# dimensioni/posizione esplicite: su WSLg i default (1920x1080) possono finire fuori schermo
	vis.create_window(window_name=percorso, width=1280, height=720, left=100, top=100)
	vis.add_geometry(mesh)

	if wireframe:
		linee = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
		linee.paint_uniform_color([0, 0, 0])
		vis.add_geometry(linee)

	render_opt = vis.get_render_option()
	render_opt.mesh_show_back_face = True
	render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Color  # forza l'uso dei colori vertice

	def salva_screenshot(vis):
		timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		nome_base = os.path.splitext(os.path.basename(percorso))[0]
		percorso_output = os.path.join(CARTELLA_SCREENSHOT, f"{nome_base}_{timestamp}.png")
		vis.capture_screen_image(percorso_output, do_render=True)
		print(f"Screenshot salvato in: {percorso_output}")
		return False  # False = non richiede un altro render extra

	vis.register_key_callback(ord("S"), salva_screenshot)

	vis.run()
	vis.destroy_window()


if __name__ == "__main__":
	visualizza(PERCORSO_MESH, wireframe=MOSTRA_WIREFRAME)