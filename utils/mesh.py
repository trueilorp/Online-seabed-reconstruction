"""
Funzioni condivise di ricostruzione 3D da point cloud (sonar/LiDAR)
====================================================================

Libreria usata da utils/pipeline.py: caricamento e filtri del CSV, pulizia della
point cloud, stima delle normali e top_surface (il fondale di una ground truth
chiusa). Le metriche sono in evaluate.py; la ricostruzione vera e propria
(specifica per ciascun algoritmo) vive negli script in scripts/.

Input: un CSV con colonne x,y,z (,intensity opzionale): scansioni sintetiche
(utils/simulate_multibeam_scans.py), Unity o sonar reale.

Dipendenze:
	pip install open3d numpy pandas
"""

import numpy as np
import open3d as o3d
import pandas as pd


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
	"""voxel_size=None salta il downsampling, nb_neighbors=None salta la rimozione degli outlier."""
	if voxel_size is None:
		pcd_ds = pcd
		print(f"[2/5] Downsampling disattivato: {len(pcd_ds.points)} punti")
	else:
		pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
		print(f"[2/5] Dopo downsampling (voxel={voxel_size}): {len(pcd_ds.points)} punti")

	if nb_neighbors is None:
		print("[2/5] Rimozione outlier disattivata")
		return pcd_ds

	pcd_clean, indici_validi = pcd_ds.remove_statistical_outlier(
		nb_neighbors=nb_neighbors, std_ratio=std_ratio
	)
	n_rimossi = len(pcd_ds.points) - len(pcd_clean.points)
	print(f"[2/5] Rimossi {n_rimossi} outlier -> {len(pcd_clean.points)} punti rimasti")

	return pcd_clean


def stima_normali(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
	"""Normali per Poisson: piano tangente sui 30 vicini piu' prossimi (default Open3D, indipendente
	dalla densita': un raggio fisso di 0.1 m col downsampling a 0.5 m non trovava vicini e lasciava
	tutte le normali al default (0,0,1)), poi orientate verso l'alto (+Y): per un fondale visto dall'alto
	"fuori" e' sempre l'acqua, piu' robusto della propagazione MST (orient_normals_consistent_tangent_plane)."""
	pcd.estimate_normals()
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
