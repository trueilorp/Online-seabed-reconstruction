"""
Entry point della pipeline di ricostruzione 3D da point cloud sonar.

Legge config/config.yaml, dove la chiave "algorithm" seleziona quale
ricostruzione eseguire ("poisson", "point2mesh", "ppsurf" o "sa_convonet"). Il
preprocessing (caricamento CSV, pulizia, normali) e la valutazione finale
sono condivisi da tutti gli algoritmi; ogni algoritmo usa solo la propria
sezione di config per i parametri specifici.

Uso:
	python reconstruct_single_dataset.py
"""

import os

import open3d as o3d

from utils.mesh import (
	valuta_mesh,
	stampa_metriche,
	salva_metriche_csv,
	confronta_con_ground_truth,
)
from utils.pipeline import (
	load_config,
	ricostruisci,
	raccogli_parametri,
	voxel_da_config,
)


def main():
	config = load_config()

	# con sliding_window abilitato point_cloud_data e' il campione tenuto dalle finestre
	mesh, point_cloud_data, tempo_calcolo, nome_metodo = ricostruisci(config)

	paths = config["paths"]
	pre = config["preprocessing"]
	valutazione = config["valutazione"]
	suffisso = "_sw" if config.get("sliding_window", {}).get("abilitato") else ""

	nome_dataset = os.path.splitext(os.path.basename(paths["input_csv"]))[0]
	cartella_output = os.path.join(paths["output_mesh_dir"], nome_dataset)
	os.makedirs(cartella_output, exist_ok=True)
	percorso_output = os.path.join(cartella_output, f"mesh_{nome_metodo}{suffisso}.ply")
	o3d.io.write_triangle_mesh(percorso_output, mesh)
	print(f"\nMesh salvata in: {percorso_output}")

	metriche = valuta_mesh(
		mesh, point_cloud_data, tempo_calcolo=tempo_calcolo, nome_metodo=nome_metodo,
		max_punti_fedelta=valutazione["max_punti_fedelta"],
		calcola_autointersezioni=valutazione["calcola_autointersezioni"],
	)
	stampa_metriche(metriche)

	if paths["ground_truth_mesh"]:
		metriche_gt = confronta_con_ground_truth(mesh, paths["ground_truth_mesh"])
		metriche.update({f"gt_{k}": v for k, v in metriche_gt.items()})

	salva_metriche_csv(
		metriche, percorso_csv=paths["metriche_csv"],
		input_csv=paths["input_csv"], voxel_size=voxel_da_config(config),
		escludi_classi=pre["escludi_classi"], parametri=raccogli_parametri(config),
	)

	if config["visualizza"]:
		o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
	main()
