"""
Entry point della pipeline di ricostruzione 3D da point cloud sonar.

Legge config/config.yaml, dove la chiave "algorithm" seleziona quale
ricostruzione eseguire ("poisson", "point2mesh", "ppsurf" o "sa_convonet"). Il
preprocessing (caricamento CSV, pulizia, normali) e la valutazione finale
sono condivisi da tutti gli algoritmi; ogni algoritmo usa solo la propria
sezione di config per i parametri specifici.

Uso:
	python main.py
"""

import os
import time

import open3d as o3d
import yaml

from utils.mesh import (
	carica_point_cloud,
	pulisci_point_cloud,
	stima_normali,
	valuta_mesh,
	stampa_metriche,
	salva_metriche_csv,
	confronta_con_ground_truth,
)
from scripts.poisson import (
	ricostruisci_poisson,
	pulisci_mesh,
)
from scripts.point2mesh import (
	subsample_aggressivo,
	ricostruisci_point2mesh,
	stampa_curva_loss,
)
from scripts.ppsurf import (
	sottocampiona_se_necessario,
	ricostruisci_ppsurf,
)
from scripts.sa_convonet import ricostruisci_sa_convonet

CONFIG_PATH = "config/config.yaml"


def load_config(percorso: str = CONFIG_PATH) -> dict:
	with open(percorso, "r") as file:
		return yaml.safe_load(file)


def carica_e_pulisci_point_cloud(config: dict) -> o3d.geometry.PointCloud:
	"""Caricamento + pulizia: comune a tutti gli algoritmi."""
	paths = config["paths"]
	pre = config["preprocessing"]
	escludi_zona_x = tuple(pre["escludi_zona_x"]) if pre["escludi_zona_x"] else None
	escludi_zona_z = tuple(pre["escludi_zona_z"]) if pre["escludi_zona_z"] else None

	pcd = carica_point_cloud(
		paths["input_csv"],
		classi_da_escludere=pre["escludi_classi"],
		altitude_min=pre["altitude_min"],
		altitude_max=pre["altitude_max"],
		escludi_zona_x=escludi_zona_x,
		escludi_zona_z=escludi_zona_z,
	)
	pcd = pulisci_point_cloud(
		pcd,
		voxel_size=pre["voxel_size"],
		nb_neighbors=pre["nb_neighbors"],
		std_ratio=pre["std_ratio"],
	)
	return pcd


def stima_normali_da_config(pcd: o3d.geometry.PointCloud, config: dict) -> o3d.geometry.PointCloud:
	pre = config["preprocessing"]
	return stima_normali(
		pcd,
		raggio_ricerca=pre["raggio_ricerca_normali"],
		max_vicini=pre["max_vicini_normali"],
	)


def esegui_poisson(pcd: o3d.geometry.PointCloud, config: dict):
	p = config["poisson"]
	pcd = stima_normali_da_config(pcd, config)

	inizio = time.time()
	mesh = ricostruisci_poisson(pcd, profondita=p["profondita"],
								 rimuovi_bassa_densita=p["rimuovi_bassa_densita"])
	tempo_calcolo = time.time() - inizio

	mesh = pulisci_mesh(mesh, dimensione_min_componente=p["dimensione_min_componente"])
	mesh.compute_vertex_normals()
	return mesh, pcd, tempo_calcolo, "poisson"


def esegui_point2mesh(pcd: o3d.geometry.PointCloud, config: dict):
	p2m = config["point2mesh"]
	paths = config["paths"]

	pcd = subsample_aggressivo(pcd, p2m["max_punti"])
	pcd = stima_normali_da_config(pcd, config)

	nome_run = os.path.splitext(os.path.basename(paths["input_csv"]))[0]
	point2mesh_dir = os.path.join(paths["repo_esterni"], "point2mesh")
	manifold_dir = os.path.join(paths["repo_esterni"], "Manifold", "build")

	mesh, tempo_calcolo, loss_per_iterazione = ricostruisci_point2mesh(
		pcd, nome_run,
		point2mesh_dir=point2mesh_dir, manifold_dir=manifold_dir,
		conda_env=p2m["conda_env"], work_dir=p2m["work_dir"],
		iterations=p2m["iterations"],
		num_facce_mesh_iniziale=p2m["initial_mesh_faces"],
		manifold_res=p2m["manifold_res"],
		export_interval=p2m["export_interval"],
	)
	stampa_curva_loss(loss_per_iterazione)
	return mesh, pcd, tempo_calcolo, "point2mesh"


def esegui_ppsurf(pcd: o3d.geometry.PointCloud, config: dict):
	pps = config["ppsurf"]
	paths = config["paths"]

	pcd = sottocampiona_se_necessario(pcd, pps["max_punti"])
	pcd = stima_normali_da_config(pcd, config)  # non usato da PPSurf, mantenuto per coerenza col preprocessing

	nome_run = os.path.splitext(os.path.basename(paths["input_csv"]))[0]
	ppsurf_dir = os.path.join(paths["repo_esterni"], "ppsurf")

	mesh, tempo_calcolo = ricostruisci_ppsurf(
		pcd, nome_run,
		ppsurf_dir=ppsurf_dir, conda_env=pps["conda_env"], work_dir=pps["work_dir"],
		gen_resolution_global=pps["gen_resolution_global"],
		rec_batch_size=pps["rec_batch_size"],
	)
	return mesh, pcd, tempo_calcolo, "ppsurf"


def esegui_sa_convonet(pcd: o3d.geometry.PointCloud, config: dict):
	sac = config["sa_convonet"]
	paths = config["paths"]

	pcd = stima_normali_da_config(pcd, config)

	nome_run = os.path.splitext(os.path.basename(paths["input_csv"]))[0]
	sa_convonet_dir = os.path.join(paths["neural_methods"], "SA-ConvONet")

	mesh, tempo_calcolo = ricostruisci_sa_convonet(
		pcd, nome_run,
		sa_convonet_dir=sa_convonet_dir, conda_env=sac["conda_env"], work_dir=sac["work_dir"],
		config_file=sac["config_file"],
		unit_size=sac["unit_size"], query_vol_size=sac["query_vol_size"],
	)
	return mesh, pcd, tempo_calcolo, "sa_convonet"


ESECUTORI = {
	"poisson": esegui_poisson,
	"point2mesh": esegui_point2mesh,
	"ppsurf": esegui_ppsurf,
	"sa_convonet": esegui_sa_convonet,
}


def main():
	config = load_config()
	algorithm = config["algorithm"]

	point_cloud_data = carica_e_pulisci_point_cloud(config)
	mesh, point_cloud_data, tempo_calcolo, nome_metodo = ESECUTORI[algorithm](point_cloud_data, config)

	paths = config["paths"]
	pre = config["preprocessing"]
	valutazione = config["valutazione"]

	os.makedirs(paths["output_mesh_dir"], exist_ok=True)
	percorso_output = os.path.join(paths["output_mesh_dir"], f"mesh_{nome_metodo}.ply")
	o3d.io.write_triangle_mesh(percorso_output, mesh)
	print(f"\nMesh salvata in: {percorso_output}")

	metriche = valuta_mesh(
		mesh, point_cloud_data, tempo_calcolo=tempo_calcolo, nome_metodo=nome_metodo,
		max_punti_fedelta=valutazione["max_punti_fedelta"],
		calcola_autointersezioni=valutazione["calcola_autointersezioni"],
	)
	stampa_metriche(metriche)
	salva_metriche_csv(
		metriche, percorso_csv=paths["metriche_csv"],
		input_csv=paths["input_csv"], voxel_size=pre["voxel_size"],
		escludi_classi=pre["escludi_classi"],
	)

	if paths["ground_truth_mesh"]:
		confronta_con_ground_truth(mesh, paths["ground_truth_mesh"])

	if config["visualizza"]:
		o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
	main()
