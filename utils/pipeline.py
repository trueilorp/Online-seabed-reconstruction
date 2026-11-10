"""
Pipeline di ricostruzione 3D usata da evaluate.py e dagli script di benchmark/.

Contiene il caricamento della config, il preprocessing comune (caricamento CSV,
pulizia) e gli esecutori dei singoli algoritmi (Poisson, PPSurf), selezionabili tramite
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
from scripts.ppsurf import (
	sottocampiona_se_necessario,
	ricostruisci_ppsurf,
)

CONFIG_PATH = "config/config.yaml"


def load_config(percorso: str = CONFIG_PATH) -> dict:
	with open(percorso, "r") as file:
		return yaml.safe_load(file)


def filtri_da_config(config: dict) -> dict:
	"""Filtri per i dati sonar reali (classi, quota, zone escluse). Opzionali in config:
	sui terreni sintetici non servono e la sezione preprocessing non li contiene."""
	pre = config["preprocessing"]
	zona_x, zona_z = pre.get("escludi_zona_x"), pre.get("escludi_zona_z")
	return dict(
		classi_da_escludere=pre.get("escludi_classi") or [],
		altitude_min=pre.get("altitude_min"),
		altitude_max=pre.get("altitude_max"),
		escludi_zona_x=tuple(zona_x) if zona_x else None,
		escludi_zona_z=tuple(zona_z) if zona_z else None,
	)


def voxel_da_config(config: dict):
	"""Lato del voxel di downsampling in metri, None se il downsampling e' disattivato."""
	ds = config["preprocessing"]["downsampling"]
	return ds["voxel_size"] if ds["abilitato"] else None


def pulisci_da_config(pcd: o3d.geometry.PointCloud, config: dict) -> o3d.geometry.PointCloud:
	outlier = config["preprocessing"]["rimozione_outlier"]
	return pulisci_point_cloud(
		pcd,
		voxel_size=voxel_da_config(config),
		nb_neighbors=outlier["nb_neighbors"] if outlier["abilitato"] else None,
		std_ratio=outlier["std_ratio"],
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
	"""Config per un terreno sintetico: input = <id>_scan.csv, algoritmo = metodo."""
	config = copy.deepcopy(config_base)
	config["paths"]["input_csv"] = percorso_scan
	config["paths"].pop("nome_run", None)
	config["algorithm"] = metodo
	return config


def esegui_poisson(pcd: o3d.geometry.PointCloud, config: dict):
	p = config["poisson"]
	pcd = stima_normali(pcd)

	inizio = time.time()
	mesh = ricostruisci_poisson(pcd, profondita=p["profondita"],
								 rimuovi_bassa_densita=p["rimuovi_bassa_densita"])
	tempo_calcolo = time.time() - inizio

	mesh = pulisci_mesh(mesh, dimensione_min_componente=p["dimensione_min_componente"])
	# Poisson interpola i colori della point cloud (intensita'/altezza): mesh salvate senza colore
	mesh.vertex_colors = o3d.utility.Vector3dVector()
	mesh.compute_vertex_normals()
	return mesh, pcd, tempo_calcolo, "poisson"

def esegui_ppsurf(pcd: o3d.geometry.PointCloud, config: dict):
	pps = config["ppsurf"]
	paths = config["paths"]

	pcd = sottocampiona_se_necessario(pcd, pps["max_punti"])

	nome_run = nome_run_da_config(config)
	ppsurf_dir = os.path.join(paths["neural_methods"], "ppsurf")

	mesh, tempo_calcolo = ricostruisci_ppsurf(
		pcd, nome_run,
		ppsurf_dir=ppsurf_dir, conda_env=pps["conda_env"], work_dir=pps["work_dir"],
		gen_resolution_global=pps["gen_resolution_global"],
		rec_batch_size=pps["rec_batch_size"],
		timeout_s=pps.get("timeout_s"),
		gen_subsample_manifold_iter=pps.get("gen_subsample_manifold_iter"),
		gen_refine_iter=pps.get("gen_refine_iter"),
	)
	return mesh, pcd, tempo_calcolo, "ppsurf"


ESECUTORI = {
	"poisson": esegui_poisson,
	"ppsurf": esegui_ppsurf,
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


def ritaglia_lontano_dai_punti(mesh: o3d.geometry.TriangleMesh, pcd: o3d.geometry.PointCloud,
							   distanza_max) -> o3d.geometry.TriangleMesh:
	"""Toglie i triangoli il cui baricentro dista piu' di distanza_max dal punto piu' vicino di pcd
	(superficie che il metodo ha inventato lontano dai dati). None = nessun ritaglio."""
	if not distanza_max or len(mesh.triangles) == 0:
		return mesh
	vertici = np.asarray(mesh.vertices)
	baricentri = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertici[np.asarray(mesh.triangles)].mean(axis=1)))
	mesh.remove_triangles_by_mask(np.asarray(baricentri.compute_point_cloud_distance(pcd)) > distanza_max)
	mesh.remove_unreferenced_vertices()
	return mesh


def distanza_max_da_config(config: dict):
	return config.get("postprocessing", {}).get("distanza_max_punti")


def _ritaglia_mesh(mesh: o3d.geometry.TriangleMesh, propri: o3d.geometry.PointCloud,
				   altri: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
	"""Tiene solo i triangoli che ricadono nella zona propria della finestra (evita superfici doppie)."""
	vertici = np.asarray(mesh.vertices)
	baricentri = vertici[np.asarray(mesh.triangles)].mean(axis=1)
	mesh.remove_triangles_by_mask(~_vicini_ai_propri(baricentri, propri, altri))
	mesh.remove_unreferenced_vertices()
	return mesh


def ricostruisci_a_finestre(config: dict, statistiche_finestre: list = None):
	"""
	Ricostruzione sliding window: ogni finestra di ping viene filtrata, pulita, ricostruita
	con l'algoritmo scelto e ritagliata sulla propria zona; le parti vengono concatenate.
	Della point cloud resta solo un piccolo campione per finestra.

	Se statistiche_finestre e' una lista, ci aggiunge un dict per finestra con:
	  latenza_s          da quando la finestra e' completa a quando la sua mesh e' unita al resto
	                     (filtri, pulizia, ricostruzione con avvio del metodo, ritaglio)
	  tempo_metodo_s     solo la ricostruzione, come misurata dall'esecutore
	  acquisizione_s     durata della finestra: dal primo all'ultimo ping (colonna elapsed_s)
	  budget_s           tempo tra il completamento della finestra precedente e di questa, cioe'
	                     quanto si ha per elaborarla prima che arrivi la successiva (per la prima
	                     finestra = acquisizione_s)
	  rtf                real-time factor = latenza_s / budget_s: < 1 tiene il passo col sonar
	"""
	sw = config["sliding_window"]
	metodo = config["algorithm"]
	filtri = filtri_da_config(config)
	voxel = voxel_da_config(config)
	nome_base = nome_run_da_config(config)
	rng = np.random.default_rng(0)

	mesh_totale = o3d.geometry.TriangleMesh()
	campioni, tempo_totale, nome_metodo = [], 0.0, metodo
	t_fine_precedente = None
	finestre = finestre_di_ping(config["paths"]["input_csv"], sw["ping_per_finestra"],
								sw["ping_sovrapposizione"], sw["righe_per_blocco"])
	for k, (df, ping_propri, col_ping) in enumerate(finestre, 1):
		inizio = time.perf_counter()
		print(f"\n--- Finestra {k}: ping {df[col_ping].iloc[0]}-{df[col_ping].iloc[-1]}, {len(df)} punti ---")
		colonne_lower = {c.lower(): c for c in df.columns}
		col_t = colonne_lower.get("elapsed_s")
		t_inizio = float(df[col_t].min()) if col_t else None
		t_fine = float(df[col_t].max()) if col_t else None
		stat = {"finestra": k, "ping_inizio": int(df[col_ping].iloc[0]), "ping_fine": int(df[col_ping].iloc[-1]),
				"n_punti": len(df), "n_punti_puliti": None, "tempo_metodo_s": None, "esito": "ok"}

		def registra(esito: str = "ok"):
			if statistiche_finestre is None:
				return
			latenza = time.perf_counter() - inizio
			acquisizione = t_fine - t_inizio if col_t else None
			budget = (t_fine - (t_fine_precedente if t_fine_precedente is not None else t_inizio)) if col_t else None
			statistiche_finestre.append({**stat, "esito": esito, "latenza_s": latenza, "acquisizione_s": acquisizione,
										 "budget_s": budget, "rtf": latenza / budget if budget else None})

		colonne = colonne_point_cloud(df)
		df = filtra_dataframe(df, colonne, **filtri)
		if df.empty:
			print("  finestra vuota dopo i filtri: saltata")
			registra("vuota")
			t_fine_precedente = t_fine
			continue
		pcd = dataframe_a_point_cloud(df, colonne)
		propri = df[col_ping].isin(ping_propri).to_numpy()
		punti = np.asarray(pcd.points)
		pcd_propri = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(punti[propri]))
		pcd_altri = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(punti[~propri]))
		if voxel:  # solo per velocizzare il ritaglio
			pcd_propri, pcd_altri = pcd_propri.voxel_down_sample(voxel), pcd_altri.voxel_down_sample(voxel)

		pcd = pulisci_da_config(pcd, config)
		stat["n_punti_puliti"] = len(pcd.points)
		if len(pcd.points) < MIN_PUNTI_FINESTRA:
			print(f"  solo {len(pcd.points)} punti dopo la pulizia: finestra saltata")
			registra("pochi_punti")
			t_fine_precedente = t_fine
			continue

		config_finestra = {**config, "paths": {**config["paths"], "nome_run": f"{nome_base}_w{k:03d}"}}
		mesh, pcd_usata, tempo, nome_metodo = ESECUTORI[metodo](pcd, config_finestra)
		tempo_totale += tempo
		stat["tempo_metodo_s"] = tempo
		mesh = _ritaglia_mesh(mesh, pcd_propri, pcd_altri)
		# distanza dai punti di QUESTA finestra: Poisson chiude ogni finestra con una cupola nell'acqua
		mesh_totale += ritaglia_lontano_dai_punti(mesh, pcd_propri + pcd_altri, distanza_max_da_config(config))
		registra()
		t_fine_precedente = t_fine

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


def ricostruisci(config: dict, statistiche_finestre: list = None):
	"""Ricostruzione completa secondo config: intera point cloud o sliding window. Restituisce (mesh, pcd, tempo, nome_metodo).
	Con la sliding window, statistiche_finestre (se e' una lista) riceve latenza e tempi di ogni finestra."""
	if config.get("sliding_window", {}).get("abilitato"):
		return ricostruisci_a_finestre(config, statistiche_finestre)
	pcd = carica_e_pulisci_point_cloud(config)
	mesh, pcd_usata, tempo, nome = ESECUTORI[config["algorithm"]](pcd, config)
	# stesso ritaglio della sliding window, cosi' i due modi restano confrontabili
	mesh = ritaglia_lontano_dai_punti(mesh, pcd, distanza_max_da_config(config))
	return mesh, pcd_usata, tempo, nome
