"""
Valutazione dei metodi di ricostruzione sui terreni sintetici.

Per ogni terreno in data/synthetic/<split>/<id>/ ricostruisce la mesh da
<id>_scan.csv con la stessa pipeline di reconstruct_single_dataset.py (utils/pipeline.py) e la confronta con
<id>_ground_truth.obj:
  - accuratezza : distanza dei punti campionati sulla mesh ricostruita dalla GT
  - completezza : distanza dei punti campionati sul fondale GT dalla mesh ricostruita
  - copertura   : frazione di punti del fondale GT entro --tau dalla mesh ricostruita
  - chamfer     : (distanza media ricostruzione->fondale + fondale->ricostruzione) / 2, in metri
  - precisione  : frazione di punti della ricostruzione entro --tau dal fondale
  - richiamo    : frazione di punti del fondale entro --tau dalla ricostruzione (= copertura)
  - fscore      : media armonica di precisione e richiamo
La GT e' chiusa (pareti e fondo): completezza, copertura, chamfer e F-score usano solo il fondale
(top_surface), altrimenti pareti e fondo, mai visti dal sonar, peserebbero come mancanti.
L'accuratezza invece e' misurata contro la GT intera.

Percorsi e default (split, limit, metodi, n_campioni, tau) nella sezione "evaluation"
di config/config.yaml; split, limit, metodi e tau sovrascrivibili da riga di comando.

Output in <evaluation.output_dir>/<split>/:
  meshes/<id>_<metodo>[_sw].ply, metriche_per_terreno.csv, metriche_per_famiglia.csv
I due CSV si accumulano tra esecuzioni (colonna "run" = istante di avvio); le mesh
invece vengono sovrascritte dall'ultima esecuzione.
Con sliding_window.abilitato in config la ricostruzione avviene a finestre di ping (suffisso _sw).

Uso:
	python evaluate.py --split val --limit 2 --methods poisson ppsurf   # 2 terreni per famiglia
"""

import argparse
import datetime
import os
import traceback

import numpy as np
import open3d as o3d
import pandas as pd

from utils.mesh import top_surface
from utils.pipeline import load_config, config_per_terreno, ricostruisci, ESECUTORI


def distanze_da_mesh(punti: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
	"""Distanza esatta punto-superficie (non punto-punto) tramite RaycastingScene."""
	scena = o3d.t.geometry.RaycastingScene()
	scena.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	return scena.compute_distance(o3d.core.Tensor(punti.astype(np.float32))).numpy()


def metriche_superficie(d_rec: np.ndarray, d_fondale: np.ndarray, tau: float) -> dict:
	"""Chamfer e F-score contro il fondale.
	d_rec: distanze dei punti della ricostruzione dal fondale; d_fondale: dei punti del fondale dalla ricostruzione."""
	precisione = float((d_rec <= tau).mean())
	richiamo = float((d_fondale <= tau).mean())
	return {
		"chamfer": float((d_rec.mean() + d_fondale.mean()) / 2),
		"precisione": precisione,
		"richiamo": richiamo,
		"fscore": 2 * precisione * richiamo / (precisione + richiamo) if precisione + richiamo > 0 else 0.0,
	}


def calcola_metriche(mesh: o3d.geometry.TriangleMesh, gt_mesh: o3d.geometry.TriangleMesh,
					 fondale: o3d.geometry.TriangleMesh, tau: float, n_campioni: int) -> dict:
	punti_mesh = np.asarray(mesh.sample_points_uniformly(n_campioni).points)
	punti_fondale = np.asarray(fondale.sample_points_uniformly(n_campioni).points)

	acc = distanze_da_mesh(punti_mesh, gt_mesh)
	comp = distanze_da_mesh(punti_fondale, mesh)
	acc_fondale = distanze_da_mesh(punti_mesh, fondale)
	return {
		"accuratezza_media": float(acc.mean()),
		"accuratezza_p90": float(np.percentile(acc, 90)),
		"completezza_media": float(comp.mean()),
		"completezza_p90": float(np.percentile(comp, 90)),
		"copertura": float((comp <= tau).mean()),
		**metriche_superficie(acc_fondale, comp, tau),
	}


def elenca_terreni(synthetic_dir: str, split: str, famiglie: dict, limit: int = None) -> list:
	"""Terreni dello split in ordine alfabetico; con limit, i primi limit di OGNI famiglia."""
	cartella = os.path.join(synthetic_dir, split)
	ids = sorted(d for d in os.listdir(cartella) if os.path.isdir(os.path.join(cartella, d)))
	if not limit:
		return ids
	per_famiglia = {}
	for id_terreno in ids:
		per_famiglia.setdefault(famiglia_di(id_terreno, famiglie), []).append(id_terreno)
	return [i for gruppo in per_famiglia.values() for i in gruppo[:limit]]


def famiglia_di(id_terreno: str, famiglie: dict) -> str:
	"""Famiglia dal manifest, altrimenti dal prefisso dell'id (flat_sand_012 -> flat_sand)."""
	return famiglie.get(id_terreno, id_terreno.rsplit("_", 1)[0])


def famiglie_da_manifest(synthetic_dir: str) -> dict:
	manifest = os.path.join(synthetic_dir, "manifest.csv")
	if not os.path.exists(manifest):
		return {}
	df = pd.read_csv(manifest)
	return dict(zip(df["id"], df["family"]))


CSV_TERRENO = "metriche_per_terreno.csv"
CSV_FAMIGLIA = "metriche_per_famiglia.csv"


def leggi_csv_precedente(percorso: str) -> pd.DataFrame:
	"""Righe delle esecuzioni precedenti (vuoto se il file non c'e')."""
	return pd.read_csv(percorso) if os.path.exists(percorso) else pd.DataFrame()


def _accoda(precedenti: pd.DataFrame, nuove: pd.DataFrame) -> pd.DataFrame:
	tutte = pd.concat([precedenti, nuove], ignore_index=True)
	return tutte[["run"] + [c for c in tutte.columns if c != "run"]]  # run sempre prima colonna


def salva_csv(righe: list, cartella_out: str, precedenti_terreno: pd.DataFrame,
			  precedenti_famiglia: pd.DataFrame):
	"""Righe di questa esecuzione in coda a quelle precedenti; le medie per famiglia sono per singola run."""
	df = pd.DataFrame(righe)
	_accoda(precedenti_terreno, df).to_csv(os.path.join(cartella_out, CSV_TERRENO), index=False, float_format="%.3f")

	ok = df[df["errore"].isna()] if "errore" in df else df
	if ok.empty:
		return
	colonne = ["accuratezza_media", "accuratezza_p90", "completezza_media", "completezza_p90",
			   "copertura", "chamfer", "precisione", "richiamo", "fscore", "n_vertici", "tempo_calcolo_secondi"]
	gruppi = ["run", "famiglia", "metodo", "sliding_window"]
	per_famiglia = ok.groupby(gruppi)[colonne].mean()
	per_famiglia.insert(0, "n_terreni", ok.groupby(gruppi).size())
	_accoda(precedenti_famiglia, per_famiglia.reset_index()).to_csv(
		os.path.join(cartella_out, CSV_FAMIGLIA), index=False, float_format="%.3f")


def main():
	config_base = load_config()
	ev = config_base["evaluation"]

	parser = argparse.ArgumentParser(description="Valuta i metodi di ricostruzione sui terreni sintetici.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test"])
	parser.add_argument("--limit", type=int, default=ev["limit"], help="numero massimo di terreni per famiglia")
	parser.add_argument("--methods", nargs="+", default=ev["metodi"], choices=list(ESECUTORI))
	parser.add_argument("--tau", type=float, default=ev["tau"], help="soglia di copertura in metri")
	args = parser.parse_args()

	sliding_window = bool(config_base.get("sliding_window", {}).get("abilitato"))
	suffisso = "_sw" if sliding_window else ""
	cartella_out = os.path.join(ev["output_dir"], args.split)
	cartella_mesh = os.path.join(cartella_out, "meshes")
	os.makedirs(cartella_mesh, exist_ok=True)

	famiglie = famiglie_da_manifest(ev["synthetic_dir"])
	terreni = elenca_terreni(ev["synthetic_dir"], args.split, famiglie, args.limit)
	righe = []
	# i CSV si accumulano tra esecuzioni: ogni run e' identificata dall'istante di avvio
	run = datetime.datetime.now().isoformat(timespec="seconds")
	precedenti_terreno = leggi_csv_precedente(os.path.join(cartella_out, CSV_TERRENO))
	precedenti_famiglia = leggi_csv_precedente(os.path.join(cartella_out, CSV_FAMIGLIA))

	# in sequenza: i metodi neurali condividono la work_dir
	for i, id_terreno in enumerate(terreni, 1):
		cartella = os.path.join(ev["synthetic_dir"], args.split, id_terreno)
		percorso_scan = os.path.join(cartella, f"{id_terreno}_scan.csv")
		percorso_gt = os.path.join(cartella, f"{id_terreno}_ground_truth.obj")
		famiglia = famiglia_di(id_terreno, famiglie)

		gt_mesh = o3d.io.read_triangle_mesh(percorso_gt)
		fondale = top_surface(gt_mesh)

		for metodo in args.methods:
			print(f"\n=== [{i}/{len(terreni)}] {id_terreno} - {metodo} ===")
			riga = {"run": run, "id": id_terreno, "famiglia": famiglia, "metodo": metodo, "sliding_window": sliding_window}
			try:
				config = config_per_terreno(config_base, percorso_scan, metodo)
				mesh, _, tempo, nome = ricostruisci(config)
				riga["metodo"] = nome  # es. "sa_convonet_online": distinto dalla variante senza adattamento

				mesh_out = os.path.join(cartella_mesh, f"{id_terreno}_{nome}{suffisso}.ply")
				o3d.io.write_triangle_mesh(mesh_out, mesh)

				riga.update(calcola_metriche(mesh, gt_mesh, fondale, args.tau, ev["n_campioni"]))
				riga.update({"n_vertici": len(mesh.vertices), "tempo_calcolo_secondi": tempo,
							 "mesh": mesh_out, "errore": None})
				print("  " + ", ".join(f"{k}={v:.4f}" for k, v in riga.items() if isinstance(v, float)))
			except Exception as e:
				traceback.print_exc()
				riga["errore"] = f"{type(e).__name__}: {e}"
			righe.append(riga)
			# riscritto a ogni passo: un crash non perde i risultati
			salva_csv(righe, cartella_out, precedenti_terreno, precedenti_famiglia)

	print(f"\nRisultati in: {cartella_out}")


if __name__ == "__main__":
	main()
