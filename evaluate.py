"""
Valutazione dei metodi di ricostruzione sui terreni sintetici.

Per ogni terreno in data/synthetic/<split>/<id>/ ricostruisce la mesh da
<id>_scan.csv con la stessa pipeline di main.py (pipeline.py) e la confronta con
<id>_ground_truth.obj:
  - accuratezza : distanza dei punti campionati sulla mesh ricostruita dalla GT
  - completezza : distanza dei punti campionati sul fondale GT dalla mesh ricostruita
  - copertura   : frazione di punti del fondale GT entro --tau dalla mesh ricostruita
La GT e' chiusa (pareti e fondo): completezza e copertura usano solo il fondale
(top_surface), altrimenti pareti e fondo, mai visti dal sonar, peserebbero come mancanti.

Output in output/evaluation/<split>/:
  meshes/<id>_<metodo>.ply, metriche_per_terreno.csv, metriche_per_famiglia.csv

Uso:
	python evaluate.py --split val --limit 2 --methods poisson ppsurf
"""

import argparse
import copy
import os
import traceback

import numpy as np
import open3d as o3d
import pandas as pd

from pipeline import load_config, carica_e_pulisci_point_cloud, ESECUTORI

SYNTHETIC_DIR = "data/synthetic"
OUTPUT_DIR = "output/evaluation"
METODI = ["poisson"]
N_CAMPIONI = 100_000  # punti campionati su ciascuna superficie per le distanze
TAU = 0.5             # soglia di copertura, m (pari al voxel_size del preprocessing)


def top_surface(gt_mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
	"""Solo il fondale della GT chiusa: scarta pareti e fondo, cioe' le facce che toccano la quota della base."""
	vertici = np.asarray(gt_mesh.vertices)
	triangoli = np.asarray(gt_mesh.triangles)
	alla_base = np.isclose(vertici[:, 1], vertici[:, 1].min(), atol=1e-4)
	tieni = ~alla_base[triangoli].any(axis=1)
	fondale = o3d.geometry.TriangleMesh(gt_mesh)
	fondale.remove_triangles_by_mask(~tieni)
	fondale.remove_unreferenced_vertices()
	return fondale


def distanze_da_mesh(punti: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
	"""Distanza esatta punto-superficie (non punto-punto) tramite RaycastingScene."""
	scena = o3d.t.geometry.RaycastingScene()
	scena.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	return scena.compute_distance(o3d.core.Tensor(punti.astype(np.float32))).numpy()


def calcola_metriche(mesh: o3d.geometry.TriangleMesh, gt_mesh: o3d.geometry.TriangleMesh,
					 fondale: o3d.geometry.TriangleMesh, tau: float) -> dict:
	punti_mesh = np.asarray(mesh.sample_points_uniformly(N_CAMPIONI).points)
	punti_fondale = np.asarray(fondale.sample_points_uniformly(N_CAMPIONI).points)

	acc = distanze_da_mesh(punti_mesh, gt_mesh)
	comp = distanze_da_mesh(punti_fondale, mesh)
	return {
		"accuratezza_media": float(acc.mean()),
		"accuratezza_p90": float(np.percentile(acc, 90)),
		"completezza_media": float(comp.mean()),
		"completezza_p90": float(np.percentile(comp, 90)),
		"copertura": float((comp <= tau).mean()),
	}


def config_per_terreno(percorso_scan: str, metodo: str) -> dict:
	config = copy.deepcopy(load_config())
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


def elenca_terreni(split: str, limit: int = None) -> list:
	cartella = os.path.join(SYNTHETIC_DIR, split)
	ids = sorted(d for d in os.listdir(cartella) if os.path.isdir(os.path.join(cartella, d)))
	return ids[:limit] if limit else ids


def famiglie_da_manifest() -> dict:
	manifest = os.path.join(SYNTHETIC_DIR, "manifest.csv")
	if not os.path.exists(manifest):
		return {}
	df = pd.read_csv(manifest)
	return dict(zip(df["id"], df["family"]))


def salva_csv(righe: list, cartella_out: str):
	df = pd.DataFrame(righe)
	df.to_csv(os.path.join(cartella_out, "metriche_per_terreno.csv"), index=False)

	ok = df[df["errore"].isna()] if "errore" in df else df
	if ok.empty:
		return
	colonne = ["accuratezza_media", "accuratezza_p90", "completezza_media", "completezza_p90",
			   "copertura", "n_vertici", "tempo_calcolo_secondi"]
	per_famiglia = ok.groupby(["famiglia", "metodo"])[colonne].mean()
	per_famiglia.insert(0, "n_terreni", ok.groupby(["famiglia", "metodo"]).size())
	per_famiglia.reset_index().to_csv(os.path.join(cartella_out, "metriche_per_famiglia.csv"), index=False)


def main():
	parser = argparse.ArgumentParser(description="Valuta i metodi di ricostruzione sui terreni sintetici.")
	parser.add_argument("--split", default="val", choices=["train", "val", "test"])
	parser.add_argument("--limit", type=int, default=None, help="numero massimo di terreni")
	parser.add_argument("--methods", nargs="+", default=METODI, choices=list(ESECUTORI))
	parser.add_argument("--tau", type=float, default=TAU, help="soglia di copertura in metri")
	args = parser.parse_args()

	cartella_out = os.path.join(OUTPUT_DIR, args.split)
	cartella_mesh = os.path.join(cartella_out, "meshes")
	os.makedirs(cartella_mesh, exist_ok=True)

	famiglie = famiglie_da_manifest()
	terreni = elenca_terreni(args.split, args.limit)
	righe = []

	# in sequenza: i metodi neurali condividono la work_dir
	for i, id_terreno in enumerate(terreni, 1):
		cartella = os.path.join(SYNTHETIC_DIR, args.split, id_terreno)
		percorso_scan = os.path.join(cartella, f"{id_terreno}_scan.csv")
		percorso_gt = os.path.join(cartella, f"{id_terreno}_ground_truth.obj")
		famiglia = famiglie.get(id_terreno, id_terreno.rsplit("_", 1)[0])

		gt_mesh = o3d.io.read_triangle_mesh(percorso_gt)
		fondale = top_surface(gt_mesh)

		for metodo in args.methods:
			print(f"\n=== [{i}/{len(terreni)}] {id_terreno} - {metodo} ===")
			riga = {"id": id_terreno, "famiglia": famiglia, "metodo": metodo}
			try:
				config = config_per_terreno(percorso_scan, metodo)
				pcd = carica_e_pulisci_point_cloud(config)
				mesh, _, tempo, nome = ESECUTORI[metodo](pcd, config)

				mesh_out = os.path.join(cartella_mesh, f"{id_terreno}_{nome}.ply")
				o3d.io.write_triangle_mesh(mesh_out, mesh)

				riga.update(calcola_metriche(mesh, gt_mesh, fondale, args.tau))
				riga.update({"n_vertici": len(mesh.vertices), "tempo_calcolo_secondi": tempo,
							 "mesh": mesh_out, "errore": None})
				print("  " + ", ".join(f"{k}={v:.4f}" for k, v in riga.items() if isinstance(v, float)))
			except Exception as e:
				traceback.print_exc()
				riga["errore"] = f"{type(e).__name__}: {e}"
			righe.append(riga)
			salva_csv(righe, cartella_out)  # riscritto a ogni passo: un crash non perde i risultati

	print(f"\nRisultati in: {cartella_out}")


if __name__ == "__main__":
	main()
