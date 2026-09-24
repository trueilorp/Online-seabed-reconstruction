import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib, os
matplotlib.use("Agg")  # headless: nessuna dipendenza grafica (WSL senza display)
import matplotlib.pyplot as plt

# --- Imposta qui i percorsi ---
DATASET = "Terrain_unity"  # nome del CSV di input (senza estensione): sottocartella di output/mesh e output/comparison
OUTPUT_DIR = f"output/comparison/{DATASET}"  # cartella dove salvare i PNG generati (creata se non esiste)
GROUND_TRUTH_PATH = "data/sonar/Terrain_ground_truth.obj"
METODI = {
	"poisson": f"output/mesh/{DATASET}/mesh_poisson.ply",
	"sa_convonet": f"output/mesh/{DATASET}/mesh_sa_convonet.ply",
	"ppsurf": f"output/mesh/{DATASET}/mesh_ppsurf.ply",
}
N_PUNTI_ERRORE = 100_000       # punti campionati per la heatmap/istogramma (sia da GT che dalle mesh ricostruite)
N_PUNTI_SEZIONE = 100          # risoluzione del profilo di sezione
Z_SEZIONE = None               # None = usa il centro del ground truth; oppure un valore specifico
SOGLIA_COPERTURA = 0.5         # distanza (m) sotto la quale un punto GT è considerato "coperto" dalla ricostruzione
OUTPUT_PREFIX = "comparison"
METRICHE_CSV = f"{OUTPUT_DIR}/metriche_accuratezza_completezza.csv"


def crea_scena_raycasting(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
	mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
	scene = o3d.t.geometry.RaycastingScene()
	scene.add_triangles(mesh_t)
	return scene


def carica_mesh(percorso: str) -> o3d.geometry.TriangleMesh:
	mesh = o3d.io.read_triangle_mesh(percorso)
	if len(mesh.vertices) == 0:
		raise ValueError(f"'{percorso}' non contiene vertici o non è stato trovato.")
	return mesh


# ---------------------------------------------------------------------------
# 1. Metriche: accuratezza, completezza, copertura
# ---------------------------------------------------------------------------
#
# Due direzioni di misura, che NON sono intercambiabili:
#
#   COMPLETEZZA (verita' -> ricostruzione): campiona punti sul ground truth,
#     misura quanto sono lontani dalla superficie ricostruita. Penalizza i
#     BUCHI (zone del vero terreno non coperte dalla mesh).
#
#   ACCURATEZZA (ricostruzione -> verita'): campiona punti sulla mesh
#     ricostruita, misura quanto sono lontani dal vero terreno. NON vede i
#     buchi (non campiona mai un punto dove non c'e' mesh) -- un metodo puo'
#     "barare" qui tagliando via le zone incerte, migliorando l'accuratezza
#     a scapito della completezza.
#
# Un metodo puo' essere ottimo su una e pessimo sull'altra (visto con
# Poisson molto tarato: accuratezza alta, completezza bassa per via dei
# buchi introdotti da rimuovi_bassa_densita alto). Vanno sempre riportate
# insieme, mai una sola.

def calcola_completezza(scene_metodo: o3d.t.geometry.RaycastingScene,
						 punti_gt: np.ndarray) -> np.ndarray:
	"""Per ogni punto del ground truth, distanza dalla superficie ricostruita."""
	query = o3d.core.Tensor(punti_gt, dtype=o3d.core.Dtype.Float32)
	return scene_metodo.compute_distance(query).numpy()


def calcola_accuratezza(mesh_metodo: o3d.geometry.TriangleMesh,
						 scene_gt: o3d.t.geometry.RaycastingScene,
						 n_punti: int) -> np.ndarray:
	"""Per ogni punto campionato sulla mesh ricostruita, distanza dal ground truth."""
	pcd_metodo = mesh_metodo.sample_points_uniformly(number_of_points=n_punti)
	query = o3d.core.Tensor(np.asarray(pcd_metodo.points), dtype=o3d.core.Dtype.Float32)
	return scene_gt.compute_distance(query).numpy()


def calcola_copertura_percentuale(distanze_completezza: np.ndarray, soglia: float) -> float:
	"""% di punti del ground truth che hanno una ricostruzione entro 'soglia' metri."""
	return 100.0 * np.mean(distanze_completezza <= soglia)


def calcola_tutte_le_metriche(mesh_gt, scene_gt, punti_gt_campionati, metodi_mesh_scene):
	"""
	metodi_mesh_scene: dict nome -> (mesh, scene) per ciascun metodo.
	Ritorna: (completezza_per_metodo, accuratezza_per_metodo, tabella_riassuntiva)
	"""
	completezza_per_metodo = {}
	accuratezza_per_metodo = {}
	righe_tabella = []

	for nome, (mesh, scene) in metodi_mesh_scene.items():
		distanze_completezza = calcola_completezza(scene, punti_gt_campionati)
		distanze_accuratezza = calcola_accuratezza(mesh, scene_gt, N_PUNTI_ERRORE)
		copertura_pct = calcola_copertura_percentuale(distanze_completezza, SOGLIA_COPERTURA)

		completezza_per_metodo[nome] = distanze_completezza
		accuratezza_per_metodo[nome] = distanze_accuratezza

		righe_tabella.append({
			"metodo": nome,
			"accuratezza_media": distanze_accuratezza.mean(),
			"accuratezza_rms": np.sqrt(np.mean(distanze_accuratezza ** 2)),
			"completezza_media": distanze_completezza.mean(),
			"completezza_rms": np.sqrt(np.mean(distanze_completezza ** 2)),
			"completezza_hausdorff": distanze_completezza.max(),
			f"copertura_pct_entro_{SOGLIA_COPERTURA}m": copertura_pct,
		})

	tabella = pd.DataFrame(righe_tabella)
	return completezza_per_metodo, accuratezza_per_metodo, tabella


# ---------------------------------------------------------------------------
# 2. Profilo di sezione (taglio verticale) -- invariato
# ---------------------------------------------------------------------------

def altezza_lungo_linea(scene: o3d.t.geometry.RaycastingScene, x_valori: np.ndarray,
						 z_fisso: float, y_partenza: float) -> np.ndarray:
	n = len(x_valori)
	origini = np.stack([x_valori, np.full(n, y_partenza), np.full(n, z_fisso)], axis=1)
	direzioni = np.tile([0.0, -1.0, 0.0], (n, 1))
	rays = o3d.core.Tensor(np.concatenate([origini, direzioni], axis=1), dtype=o3d.core.Dtype.Float32)

	risultato = scene.cast_rays(rays)
	t_hit = risultato["t_hit"].numpy()

	altezze = y_partenza - t_hit  # NaN/inf dove non c'è intersezione (nessun dato lì)
	altezze[np.isinf(altezze)] = np.nan
	return altezze


def genera_profilo_sezione(scene_gt, scene_metodi, bbox_gt):
	min_gt, max_gt = bbox_gt
	z_sezione = Z_SEZIONE if Z_SEZIONE is not None else (min_gt[2] + max_gt[2]) / 2
	x_valori = np.linspace(min_gt[0], max_gt[0], N_PUNTI_SEZIONE)
	y_partenza = max_gt[1] + 10  # sopra tutto, per essere sicuri di partire fuori dalla mesh

	altezze_gt = altezza_lungo_linea(scene_gt, x_valori, z_sezione, y_partenza)

	fig, ax = plt.subplots(figsize=(12, 5))
	ax.plot(x_valori, altezze_gt, linestyle=":", color="lightgray", linewidth=2.5, label="Ground truth", zorder=10)

	stili = ["--", ":", "-.", "-"]
	for (nome, scene), stile in zip(scene_metodi.items(), stili):
		altezze = altezza_lungo_linea(scene, x_valori, z_sezione, y_partenza)
		ax.plot(x_valori, altezze, stile, linewidth=1.8, label=nome)

	ax.set_xlabel("X")
	ax.set_ylabel("Altezza (Y)")
	ax.set_title(f"Profilo di sezione a Z = {z_sezione:.2f}\n"
				 f"(linee interrotte = nessun dato/nessuna superficie ricostruita in quel punto)")
	ax.legend()
	ax.grid(alpha=0.3)
	plt.tight_layout()
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	percorso_out = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_profilo_sezione.png")
	plt.savefig(percorso_out, dpi=150)
	plt.close(fig)
	print(f"Profilo di sezione salvato in: {percorso_out}")


# ---------------------------------------------------------------------------
# 3. Istogrammi -- ora uno per completezza e uno per accuratezza
# ---------------------------------------------------------------------------

def genera_istogramma(errori_per_metodo: dict, titolo: str, nome_file: str):
	fig, ax = plt.subplots(figsize=(9, 5))
	for nome, distanze in errori_per_metodo.items():
		ax.hist(distanze, bins=80, alpha=0.55, label=f"{nome} (media {distanze.mean():.3f} m)",
				 range=(0, np.percentile(distanze, 99)))
	ax.set_xlabel("Distanza (m)")
	ax.set_ylabel("Conteggio punti")
	ax.set_title(titolo)
	ax.legend()
	plt.tight_layout()
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	percorso_out = os.path.join(OUTPUT_DIR, nome_file)
	plt.savefig(percorso_out, dpi=150)
	plt.close(fig)
	print(f"Istogramma salvato in: {percorso_out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
	print(f"Carico ground truth: {GROUND_TRUTH_PATH}")
	mesh_gt = carica_mesh(GROUND_TRUTH_PATH)
	scene_gt = crea_scena_raycasting(mesh_gt)

	vertici_gt = np.asarray(mesh_gt.vertices)
	bbox_gt = (vertici_gt.min(axis=0), vertici_gt.max(axis=0))

	pcd_gt_campionata = mesh_gt.sample_points_uniformly(number_of_points=N_PUNTI_ERRORE)
	punti_gt = np.asarray(pcd_gt_campionata.points)

	metodi_mesh_scene = {}
	scene_metodi = {}  # solo per il profilo di sezione, che serve solo le scene
	for nome, percorso in METODI.items():
		print(f"Carico {nome}: {percorso}")
		mesh = carica_mesh(percorso)
		scene = crea_scena_raycasting(mesh)
		metodi_mesh_scene[nome] = (mesh, scene)
		scene_metodi[nome] = scene

	print("\nCalcolo accuratezza, completezza e copertura per ciascun metodo...")
	completezza_per_metodo, accuratezza_per_metodo, tabella = calcola_tutte_le_metriche(
		mesh_gt, scene_gt, punti_gt, metodi_mesh_scene
	)

	print("\nGenero profilo di sezione...")
	genera_profilo_sezione(scene_gt, scene_metodi, bbox_gt)

	print("\nGenero istogramma di completezza (verità -> ricostruzione, penalizza i buchi)...")
	genera_istogramma(completezza_per_metodo,
					   "Completezza: distanza dal ground truth alla ricostruzione più vicina",
					   f"{OUTPUT_PREFIX}_istogramma_completezza.png")

	print("\nGenero istogramma di accuratezza (ricostruzione -> verità, cieco ai buchi)...")
	genera_istogramma(accuratezza_per_metodo,
					   "Accuratezza: distanza dalla ricostruzione al ground truth più vicino",
					   f"{OUTPUT_PREFIX}_istogramma_accuratezza.png")

	print(f"\nFatto. Apri i PNG generati in {OUTPUT_DIR}/ da Windows/VS Code.")


if __name__ == "__main__":
	main()