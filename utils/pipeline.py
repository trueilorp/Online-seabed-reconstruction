"""
Pipeline di ricostruzione 3D condivisa da reconstruct_single_dataset.py ed evaluate.py.

Contiene il caricamento della config, il preprocessing comune (caricamento CSV,
pulizia, normali) e gli esecutori dei singoli algoritmi, selezionabili tramite
ESECUTORI[config["algorithm"]].

Punto d'ingresso per gli script: ricostruisci(config), che esegue l'algoritmo
sull'intera point cloud oppure, se sliding_window.abilitato, a finestre di ping.
"""

import copy
import os
import time

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from utils.mesh import (
	carica_point_cloud,
	colonne_point_cloud,
	filtra_dataframe,
	dataframe_a_point_cloud,
	pulisci_point_cloud,
	stima_normali,
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


def filtri_da_config(config: dict) -> dict:
	pre = config["preprocessing"]
	return dict(
		classi_da_escludere=pre["escludi_classi"],
		altitude_min=pre["altitude_min"],
		altitude_max=pre["altitude_max"],
		escludi_zona_x=tuple(pre["escludi_zona_x"]) if pre["escludi_zona_x"] else None,
		escludi_zona_z=tuple(pre["escludi_zona_z"]) if pre["escludi_zona_z"] else None,
	)


def pulisci_da_config(pcd: o3d.geometry.PointCloud, config: dict) -> o3d.geometry.PointCloud:
	pre = config["preprocessing"]
	return pulisci_point_cloud(
		pcd,
		voxel_size=pre["voxel_size"],
		nb_neighbors=pre["nb_neighbors"],
		std_ratio=pre["std_ratio"],
	)


def carica_e_pulisci_point_cloud(config: dict) -> o3d.geometry.PointCloud:
	"""Caricamento + pulizia: comune a tutti gli algoritmi."""
	pcd = carica_point_cloud(config["paths"]["input_csv"], **filtri_da_config(config))
	return pulisci_da_config(pcd, config)


def nome_run_da_config(config: dict) -> str:
	"""Nome della run per i file dei metodi neurali: il nome del CSV, o paths.nome_run se impostato (finestre)."""
	paths = config["paths"]
	return paths.get("nome_run") or os.path.splitext(os.path.basename(paths["input_csv"]))[0]


def config_per_terreno(config_base: dict, percorso_scan: str, metodo: str) -> dict:
	"""Config per un terreno sintetico: input = <id>_scan.csv, filtri dei dati reali disattivati."""
	config = copy.deepcopy(config_base)
	config["paths"]["input_csv"] = percorso_scan
	config["algorithm"] = metodo
	# i filtri del config sono pensati per i dati reali: sui sintetici non servono
	pre = config["preprocessing"]
	pre["escludi_classi"] = []
	pre["altitude_min"] = None
	pre["altitude_max"] = None
	pre["escludi_zona_x"] = None
	pre["escludi_zona_z"] = None
	return config


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

	nome_run = nome_run_da_config(config)
	point2mesh_dir = os.path.join(paths["neural_methods"], "point2mesh")
	manifold_dir = os.path.join(paths["neural_methods"], "Manifold", "build")

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

	nome_run = nome_run_da_config(config)
	ppsurf_dir = os.path.join(paths["neural_methods"], "ppsurf")

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

	nome_run = nome_run_da_config(config)
	sa_convonet_dir = os.path.join(paths["neural_methods"], "SA-ConvONet")

	mesh, tempo_calcolo = ricostruisci_sa_convonet(
		pcd, nome_run,
		sa_convonet_dir=sa_convonet_dir, conda_env=sac["conda_env"], work_dir=sac["work_dir"],
		config_file=sac["config_file"],
		unit_size=sac["unit_size"], query_vol_size=sac["query_vol_size"],
	)
	return mesh, pcd, tempo_calcolo, "sa_convonet"


# chiavi di config che non sono parametri della ricostruzione (percorsi, ambienti, logging)
CHIAVI_NON_PARAMETRI = {"conda_env", "work_dir", "export_interval", "config_file"}


def raccogli_parametri(config: dict) -> dict:
	"""Parametri della run da salvare nel CSV: preprocessing, valutazione e sezione del metodo usato."""
	def formatta(valore):
		if valore is None:
			return "nessuno"  # "null" in config (nessun limite/filtro); non "null", che pandas rilegge come cella vuota
		if isinstance(valore, (list, tuple)):
			return str(list(valore))
		return valore

	algorithm = config["algorithm"]
	parametri = {}
	for chiave, valore in config["preprocessing"].items():
		# voxel_size ed escludi_classi hanno gia' colonne dedicate (presenti anche nelle righe vecchie)
		if chiave not in ("voxel_size", "escludi_classi"):
			parametri[f"pre_{chiave}"] = formatta(valore)
	for chiave, valore in config["valutazione"].items():
		parametri[f"val_{chiave}"] = formatta(valore)
	for chiave, valore in config.get(algorithm, {}).items():
		if chiave not in CHIAVI_NON_PARAMETRI:
			parametri[f"{algorithm}_{chiave}"] = formatta(valore)
	# solo se attivo: con sliding window disabilitato la riga del CSV resta quella di prima
	sw = config.get("sliding_window", {})
	if sw.get("abilitato"):
		for chiave, valore in sw.items():
			parametri[f"sw_{chiave}"] = formatta(valore)
	return parametri


ESECUTORI = {
	"poisson": esegui_poisson,
	"point2mesh": esegui_point2mesh,
	"ppsurf": esegui_ppsurf,
	"sa_convonet": esegui_sa_convonet,
}


# ---------------------------------------------------------------------------
# Sliding window: i dati arrivano a blocchi di ping e vengono scartati dopo l'uso
# ---------------------------------------------------------------------------

MIN_PUNTI_FINESTRA = 100  # sotto questa soglia (dopo la pulizia) la finestra viene saltata


def colonna_ping(df: pd.DataFrame) -> str:
	colonne_lower = {c.lower(): c for c in df.columns}
	col = colonne_lower.get("ping_number") or colonne_lower.get("ping number")
	if col is None:
		raise ValueError(f"Sliding window: nessuna colonna ping nel CSV. Colonne: {list(df.columns)}")
	return col


def finestre_di_ping(percorso_csv: str, ping_per_finestra: int, ping_sovrapposizione: int,
					 righe_per_blocco: int):
	"""
	Legge il CSV a blocchi di righe (come arriverebbero dal sonar) e restituisce finestre
	di ping_per_finestra ping consecutivi, sovrapposte di ping_sovrapposizione ping.
	In memoria resta solo la finestra corrente: i ping usati vengono scartati.

	Restituisce (df_finestra, ping_propri, col_ping). ping_propri sono i ping di cui la
	finestra "possiede" la mesh: la sovrapposizione e' divisa a meta' tra due finestre
	consecutive, cosi' ogni zona del fondale e' assegnata a una sola finestra.
	"""
	if not 0 <= ping_sovrapposizione < ping_per_finestra:
		raise ValueError("Sliding window: serve 0 <= ping_sovrapposizione < ping_per_finestra")
	passo = ping_per_finestra - ping_sovrapposizione
	meta_sx = ping_sovrapposizione // 2
	meta_dx = ping_sovrapposizione - meta_sx

	buffer, col_ping, prima = None, None, True
	for blocco in pd.read_csv(percorso_csv, chunksize=righe_per_blocco):
		if col_ping is None:
			col_ping = colonna_ping(blocco)
		buffer = blocco if buffer is None else pd.concat([buffer, blocco], ignore_index=True)
		ping = pd.unique(buffer[col_ping])  # in ordine di arrivo
		# la finestra e' completa solo quando e' arrivato almeno un ping successivo
		while len(ping) > ping_per_finestra:
			ping_finestra = ping[:ping_per_finestra]
			inizio = 0 if prima else meta_sx
			yield (buffer[buffer[col_ping].isin(ping_finestra)],
				   ping_finestra[inizio:ping_per_finestra - meta_dx], col_ping)
			buffer = buffer[~buffer[col_ping].isin(ping[:passo])]  # dimentica i ping gia' usati
			ping = ping[passo:]
			prima = False

	if buffer is not None and len(buffer):  # ultima finestra: tiene tutto quello che resta
		yield buffer, ping[0 if prima else meta_sx:], col_ping


def _vicini_ai_propri(query: np.ndarray, propri: o3d.geometry.PointCloud,
					  altri: o3d.geometry.PointCloud) -> np.ndarray:
	"""True per i punti di query piu' vicini ai punti propri della finestra che a quelli in sovrapposizione."""
	if len(altri.points) == 0:
		return np.ones(len(query), dtype=bool)
	q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query))
	return np.asarray(q.compute_point_cloud_distance(propri)) <= np.asarray(q.compute_point_cloud_distance(altri))


def _ritaglia_mesh(mesh: o3d.geometry.TriangleMesh, propri: o3d.geometry.PointCloud,
				   altri: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
	"""Tiene solo i triangoli che ricadono nella zona propria della finestra (evita superfici doppie)."""
	vertici = np.asarray(mesh.vertices)
	baricentri = vertici[np.asarray(mesh.triangles)].mean(axis=1)
	mesh.remove_triangles_by_mask(~_vicini_ai_propri(baricentri, propri, altri))
	mesh.remove_unreferenced_vertices()
	return mesh


def ricostruisci_a_finestre(config: dict):
	"""
	Ricostruzione sliding window: ogni finestra di ping viene filtrata, pulita, ricostruita
	con l'algoritmo scelto e ritagliata sulla propria zona; le parti vengono concatenate.
	Della point cloud resta solo un piccolo campione per finestra, usato per le metriche.
	"""
	sw = config["sliding_window"]
	metodo = config["algorithm"]
	filtri = filtri_da_config(config)
	voxel = config["preprocessing"]["voxel_size"]
	nome_base = nome_run_da_config(config)
	rng = np.random.default_rng(0)

	mesh_totale = o3d.geometry.TriangleMesh()
	campioni, tempo_totale, nome_metodo = [], 0.0, metodo
	finestre = finestre_di_ping(config["paths"]["input_csv"], sw["ping_per_finestra"],
								sw["ping_sovrapposizione"], sw["righe_per_blocco"])
	for k, (df, ping_propri, col_ping) in enumerate(finestre, 1):
		print(f"\n--- Finestra {k}: ping {df[col_ping].iloc[0]}-{df[col_ping].iloc[-1]}, {len(df)} punti ---")
		colonne = colonne_point_cloud(df)
		df = filtra_dataframe(df, colonne, **filtri)
		if df.empty:
			print("  finestra vuota dopo i filtri: saltata")
			continue
		pcd = dataframe_a_point_cloud(df, colonne)
		propri = df[col_ping].isin(ping_propri).to_numpy()
		punti = np.asarray(pcd.points)
		pcd_propri = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(punti[propri]))
		pcd_altri = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(punti[~propri]))
		if voxel:  # solo per velocizzare il ritaglio
			pcd_propri, pcd_altri = pcd_propri.voxel_down_sample(voxel), pcd_altri.voxel_down_sample(voxel)

		pcd = pulisci_da_config(pcd, config)
		if len(pcd.points) < MIN_PUNTI_FINESTRA:
			print(f"  solo {len(pcd.points)} punti dopo la pulizia: finestra saltata")
			continue

		config_finestra = {**config, "paths": {**config["paths"], "nome_run": f"{nome_base}_w{k:03d}"}}
		mesh, pcd_usata, tempo, nome_metodo = ESECUTORI[metodo](pcd, config_finestra)
		tempo_totale += tempo
		mesh_totale += _ritaglia_mesh(mesh, pcd_propri, pcd_altri)

		punti_usati = np.asarray(pcd_usata.points)
		punti_usati = punti_usati[_vicini_ai_propri(punti_usati, pcd_propri, pcd_altri)]
		n = min(len(punti_usati), sw["punti_campione_per_finestra"])
		campioni.append(punti_usati[rng.choice(len(punti_usati), n, replace=False)])
		# df, pcd e mesh della finestra escono di scope qui: in memoria restano solo mesh_totale e i campioni

	if len(mesh_totale.triangles) == 0:
		raise RuntimeError("Sliding window: nessuna finestra ha prodotto una mesh")
	mesh_totale.remove_duplicated_vertices()
	mesh_totale.compute_vertex_normals()
	campione = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(campioni)))
	print(f"\nSliding window: {len(mesh_totale.vertices)} vertici, {len(campione.points)} punti tenuti per le metriche")
	return mesh_totale, campione, tempo_totale, nome_metodo


def ricostruisci(config: dict):
	"""Ricostruzione completa secondo config: intera point cloud o sliding window. Restituisce (mesh, pcd, tempo, nome_metodo)."""
	if config.get("sliding_window", {}).get("abilitato"):
		return ricostruisci_a_finestre(config)
	pcd = carica_e_pulisci_point_cloud(config)
	return ESECUTORI[config["algorithm"]](pcd, config)
