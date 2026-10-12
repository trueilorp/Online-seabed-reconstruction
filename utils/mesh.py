"""
Funzioni condivise di ricostruzione 3D da point cloud (sonar/LiDAR)
====================================================================

Libreria usata da main.py e dalle pipeline in scripts/ (poisson.py,
point2mesh.py, ppsurf.py): caricamento/pulizia della point cloud, stima
normali, valutazione e salvataggio metriche. La ricostruzione vera e
propria (specifica per ciascun algoritmo) vive negli script in scripts/.

Input:  un file CSV con colonne x,y,z (,intensity opzionale) prodotto da
		PointCloudRecorder.cs nel simulatore Unity.
Output: una mesh .ply/.obj ricostruita, più eventuali metriche di
		confronto con un ground truth (se disponibile).

Dipendenze:
	pip install open3d numpy pandas
"""

import numpy as np
import open3d as o3d
import pandas as pd

import os

def carica_point_cloud(percorso_csv: str, classi_da_escludere: list = None,
						altitude_min: float = None, altitude_max: float = None,
						escludi_zona_x: tuple = None, escludi_zona_z: tuple = None) -> o3d.geometry.PointCloud:
	df = pd.read_csv(percorso_csv)
	colonne = colonne_point_cloud(df)
	df = filtra_dataframe(df, colonne, classi_da_escludere=classi_da_escludere,
						  altitude_min=altitude_min, altitude_max=altitude_max,
						  escludi_zona_x=escludi_zona_x, escludi_zona_z=escludi_zona_z)
	pcd = dataframe_a_point_cloud(df, colonne)
	print(f"[1/5] Caricati {len(pcd.points)} punti da {percorso_csv} "
		  f"(X={colonne['x']}, Y={colonne['y']}, Z={colonne['z']})")
	return pcd


def colonne_point_cloud(df: pd.DataFrame) -> dict:
	"""Nomi delle colonne x/y/z, intensita' e classification nei vari formati CSV (Unity, Omniscan, Surveyor)."""
	colonne_lower = {c.lower(): c for c in df.columns}

	col_x = colonne_lower.get("x") or colonne_lower.get("easting (local m)")
	col_z = colonne_lower.get("z") or colonne_lower.get("northing (local m)")
	col_y = colonne_lower.get("y") or colonne_lower.get("altitude (m)")
	col_intensity = (
		colonne_lower.get("intensity")
		or colonne_lower.get("intensita")
		or colonne_lower.get("power (db)")
	)
	col_classification = colonne_lower.get("classification")

	if not (col_x and col_y and col_z):
		raise ValueError(
			f"Non trovo colonne x,y,z (o easting/northing/altitude). Colonne nel CSV: {list(df.columns)}"
		)
	return {"x": col_x, "y": col_y, "z": col_z,
			"intensity": col_intensity, "classification": col_classification}


def filtra_dataframe(df: pd.DataFrame, colonne: dict, classi_da_escludere: list = None,
					 altitude_min: float = None, altitude_max: float = None,
					 escludi_zona_x: tuple = None, escludi_zona_z: tuple = None) -> pd.DataFrame:
	"""Filtri sui dati grezzi (classi, range di altitudine, zona rettangolare esclusa)."""
	col_x, col_y, col_z = colonne["x"], colonne["y"], colonne["z"]
	col_classification = colonne["classification"]

	if classi_da_escludere and col_classification:
		n_prima = len(df)
		df = df[~df[col_classification].isin(classi_da_escludere)]
		print(f"[1/5] Filtrate {n_prima - len(df)} righe con classification in {classi_da_escludere} "
			  f"(provvisorio, da confermare) -> {len(df)} righe rimaste")

	if altitude_min is not None or altitude_max is not None:
		n_prima = len(df)
		if altitude_min is not None:
			df = df[df[col_y] >= altitude_min]
		if altitude_max is not None:
			df = df[df[col_y] <= altitude_max]
		print(f"[1/5] Filtrate {n_prima - len(df)} righe fuori dal range altitude "
			  f"[{altitude_min}, {altitude_max}] -> {len(df)} righe rimaste")

	if escludi_zona_x is not None and escludi_zona_z is not None:
		n_prima = len(df)
		dentro_zona = (
			(df[col_x] >= escludi_zona_x[0]) & (df[col_x] <= escludi_zona_x[1]) &
			(df[col_z] >= escludi_zona_z[0]) & (df[col_z] <= escludi_zona_z[1])
		)
		df = df[~dentro_zona]
		print(f"[1/5] Escluse {n_prima - len(df)} righe nella zona X{escludi_zona_x} x Z{escludi_zona_z} "
			  f"(es. sosta barca) -> {len(df)} righe rimaste")
	return df


def dataframe_a_point_cloud(df: pd.DataFrame, colonne: dict) -> o3d.geometry.PointCloud:
	"""Point cloud Open3D dalle colonne x/y/z, colorata per intensita' (o per altezza se assente)."""
	col_intensity = colonne["intensity"]
	punti = df[[colonne["x"], colonne["y"], colonne["z"]]].to_numpy(dtype=np.float64)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(punti)

	if col_intensity:
		intens = df[col_intensity].to_numpy(dtype=np.float64)
		intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-9)
		colori = np.stack([intens_norm, intens_norm, 1 - intens_norm], axis=1)
		pcd.colors = o3d.utility.Vector3dVector(colori)
		print(f"  Colorazione mesh basata su: {col_intensity}")
	else:
		# Nessuna colonna di intensità disponibile (es. Surveyor 240): senza colore
		# assegnato Open3D userebbe nero di default, rendendo la mesh invisibile a
		# qualsiasi illuminazione. Coloriamo per altezza (Y) come fallback sensato.
		altezze = punti[:, 1]
		altezze_norm = (altezze - altezze.min()) / (altezze.max() - altezze.min() + 1e-9)
		# blu (basso) -> giallo (alto), stessa logica cromatica di viridis semplificata
		colori = np.stack([altezze_norm, altezze_norm, 1 - altezze_norm], axis=1)
		pcd.colors = o3d.utility.Vector3dVector(colori)
		print(f"  Colonna intensità non trovata: colorazione mesh basata sull'altezza (Y) come fallback")
	return pcd


def pulisci_point_cloud(pcd: o3d.geometry.PointCloud,
						 voxel_size=None,
						 nb_neighbors: int = 20,
						 std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
	
	pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
	print(f"[2/5] Dopo downsampling (voxel={voxel_size}): {len(pcd_ds.points)} punti")

	pcd_clean, indici_validi = pcd_ds.remove_statistical_outlier(
		nb_neighbors=nb_neighbors, std_ratio=std_ratio
	)
	n_rimossi = len(pcd_ds.points) - len(pcd_clean.points)
	print(f"[2/5] Rimossi {n_rimossi} outlier -> {len(pcd_clean.points)} punti rimasti")

	return pcd_clean


def stima_normali(pcd: o3d.geometry.PointCloud, raggio_ricerca: float = 0.1,
				   max_vicini: int = 30) -> o3d.geometry.PointCloud:

	pcd.estimate_normals(
		search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=raggio_ricerca, max_nn=max_vicini)
	)
	pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 1.0, 0.0]))
	print("[3/5] Normali stimate e orientate")
	return pcd


def top_surface(gt_mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
	"""Solo il fondale di una GT chiusa (terreni sintetici): scarta pareti e fondo, cioe' le facce che toccano la quota della base."""
	vertici = np.asarray(gt_mesh.vertices)
	triangoli = np.asarray(gt_mesh.triangles)
	alla_base = np.isclose(vertici[:, 1], vertici[:, 1].min(), atol=1e-4)
	tieni = ~alla_base[triangoli].any(axis=1)
	fondale = o3d.geometry.TriangleMesh(gt_mesh)
	fondale.remove_triangles_by_mask(~tieni)
	fondale.remove_unreferenced_vertices()
	return fondale


# ---------------------------------------------------------------------------
# 4. METRICHE DI VALUTAZIONE (senza bisogno di ground truth esterno)
# ---------------------------------------------------------------------------

def valuta_mesh(mesh: o3d.geometry.TriangleMesh, pcd_originale: o3d.geometry.PointCloud,
				 tempo_calcolo: float = None, nome_metodo: str = "",
				 max_punti_fedelta: int = 50_000,
				 calcola_autointersezioni: bool = True) -> dict:

	metriche = {"metodo": nome_metodo}

	punti_originali = np.asarray(pcd_originale.points)
	if len(punti_originali) > max_punti_fedelta:
		indici = np.random.choice(len(punti_originali), max_punti_fedelta, replace=False)
		punti_per_fedelta = punti_originali[indici]
		print(f"  (fedeltà calcolata su un campione di {max_punti_fedelta} punti "
			  f"su {len(punti_originali)} totali, per velocità)")
	else:
		punti_per_fedelta = punti_originali

	mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
	scene = o3d.t.geometry.RaycastingScene()
	scene.add_triangles(mesh_t)

	punti_query = o3d.core.Tensor(punti_per_fedelta, dtype=o3d.core.Dtype.Float32)
	distanze = scene.compute_distance(punti_query).numpy()

	metriche["fedelta_distanza_media"] = float(np.mean(distanze))
	metriche["fedelta_distanza_rms"] = float(np.sqrt(np.mean(distanze ** 2)))
	metriche["fedelta_distanza_max"] = float(np.max(distanze))

	# --- Qualità intrinseca della mesh ---
	metriche["n_vertici"] = len(mesh.vertices)
	metriche["n_triangoli"] = len(mesh.triangles)
	metriche["watertight"] = mesh.is_watertight()
	metriche["edge_manifold"] = mesh.is_edge_manifold()
	metriche["vertex_manifold"] = mesh.is_vertex_manifold()

	if calcola_autointersezioni:
		triangoli_autointersecanti = mesh.get_self_intersecting_triangles()
		metriche["n_triangoli_autointersecanti"] = len(triangoli_autointersecanti)
		metriche["percentuale_autointersecanti"] = (
			100 * len(triangoli_autointersecanti) / max(len(mesh.triangles), 1)
		)
	else:
		metriche["n_triangoli_autointersecanti"] = None
		metriche["percentuale_autointersecanti"] = None

	metriche["area_superficie"] = mesh.get_surface_area()

	# Regolarità: area di ciascun triangolo, guardiamo dispersione (std/media = coeff. variazione)
	vertici = np.asarray(mesh.vertices)
	triangoli_np = np.asarray(mesh.triangles)
	a = vertici[triangoli_np[:, 0]]
	b = vertici[triangoli_np[:, 1]]
	c = vertici[triangoli_np[:, 2]]
	aree_triangoli = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
	metriche["area_triangolo_media"] = float(np.mean(aree_triangoli))
	metriche["coeff_variazione_aree"] = float(np.std(aree_triangoli) / (np.mean(aree_triangoli) + 1e-12))

	if tempo_calcolo is not None:
		metriche["tempo_calcolo_secondi"] = tempo_calcolo

	return metriche


def stampa_metriche(metriche: dict):
	print(f"\n{'=' * 50}")
	print(f"Metriche — {metriche.get('metodo', '')}")
	print(f"{'=' * 50}")
	print(f"Fedeltà ai dati originali (raycasting):")
	print(f"  distanza media : {metriche['fedelta_distanza_media']:.5f} m")
	print(f"  distanza RMS   : {metriche['fedelta_distanza_rms']:.5f} m")
	print(f"  distanza max   : {metriche['fedelta_distanza_max']:.5f} m")
	print(f"Qualità intrinseca:")
	print(f"  vertici/triangoli   : {metriche['n_vertici']} / {metriche['n_triangoli']}")
	print(f"  watertight          : {metriche['watertight']}")
	print(f"  edge/vertex manifold: {metriche['edge_manifold']} / {metriche['vertex_manifold']}")
	if metriche["n_triangoli_autointersecanti"] is not None:
		print(f"  triangoli auto-intersecanti: {metriche['n_triangoli_autointersecanti']} "
			  f"({metriche['percentuale_autointersecanti']:.2f}%)")
	else:
		print(f"  triangoli auto-intersecanti: (non calcolato, CALCOLA_AUTOINTERSEZIONI=False)")
	print(f"  area superficie     : {metriche['area_superficie']:.3f}")
	print(f"  area triangolo media: {metriche['area_triangolo_media']:.6f} "
		  f"(coeff. variazione: {metriche['coeff_variazione_aree']:.2f})")
	if "tempo_calcolo_secondi" in metriche:
		print(f"  tempo di calcolo    : {metriche['tempo_calcolo_secondi']:.2f} s")


# metriche calcolate e stampate da valuta_mesh ma non salvate nel CSV (poco utili nell'analisi)
COLONNE_ESCLUSE_CSV = [
	"n_triangoli", "watertight", "edge_manifold", "vertex_manifold",
	"n_triangoli_autointersecanti", "percentuale_autointersecanti",
	"area_triangolo_media", "coeff_variazione_aree",
]


def salva_metriche_csv(metriche: dict, percorso_csv: str = "../output/metriche_ricostruzione.csv",
						input_csv: str = None, voxel_size=None, escludi_classi: list = None,
						parametri: dict = None):
	import datetime

	riga = {k: v for k, v in metriche.items() if k not in COLONNE_ESCLUSE_CSV}
	riga["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
	riga["input_csv"] = input_csv
	riga["voxel_size"] = voxel_size if voxel_size is not None else "auto"
	riga["escludi_classi"] = str(escludi_classi)
	riga.update(parametri or {})

	df_riga = pd.DataFrame([riga])
	if os.path.isfile(percorso_csv):
		# letto come testo: le righe vecchie vengono riscritte identiche (niente 234231 -> 234231.0)
		df_esistente = pd.read_csv(percorso_csv, dtype=str, keep_default_na=False)
		if list(df_esistente.columns) == list(df_riga.columns):
			df_riga.to_csv(percorso_csv, mode="a", header=False, index=False)
		else:
			# colonne diverse (es. metriche ground truth aggiunte/assenti): riscrive il file con l'unione delle colonne
			pd.concat([df_esistente, df_riga.astype(object)], ignore_index=True).to_csv(percorso_csv, index=False)
	else:
		df_riga.to_csv(percorso_csv, index=False)
	print(f"Metriche aggiunte a: {percorso_csv}")

def confronta_con_ground_truth(mesh: o3d.geometry.TriangleMesh,
								percorso_gt_mesh: str) -> dict:

	gt_mesh = o3d.io.read_triangle_mesh(percorso_gt_mesh)
	gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=100_000)
	ricostruita_pcd = mesh.sample_points_uniformly(number_of_points=100_000)

	distanze = np.asarray(ricostruita_pcd.compute_point_cloud_distance(gt_pcd))

	metriche = {
		"distanza_media": float(np.mean(distanze)),
		"distanza_rms": float(np.sqrt(np.mean(distanze ** 2))),
		"distanza_hausdorff_approx": float(np.max(distanze)),
	}
	print("Metriche di confronto con ground truth:")
	for k, v in metriche.items():
		print(f"  {k}: {v:.4f}")
	return metriche