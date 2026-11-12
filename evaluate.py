"""
Valutazione dei metodi di ricostruzione sui terreni sintetici.

Per ogni terreno in data/synthetic/<split>/<id>/ ricostruisce la mesh da
<id>_scan.csv con la pipeline di utils/pipeline.py e la confronta con
<id>_ground_truth.obj:
  - accuratezza : distanza dei punti campionati sulla mesh ricostruita dalla GT
  - completezza : distanza dei punti campionati sul fondale GT dalla mesh ricostruita
  - copertura   : frazione di punti del fondale GT entro --tau dalla mesh ricostruita
  - chamfer     : (distanza media ricostruzione->fondale + fondale->ricostruzione) / 2, in metri
  - precisione  : frazione di punti della ricostruzione entro --tau dal fondale
  - richiamo    : frazione di punti del fondale entro --tau dalla ricostruzione (= copertura)
  - fscore      : media armonica di precisione e richiamo
  - rec2gt_*, gt2rec_* : mediana, iqr (p75 - p25) e p95 delle distanze ricostruzione->fondale e
                  fondale->ricostruzione: la media (Chamfer) e' dominata dalle code, queste no
Con la sliding window, per ogni finestra: latenza, tempo di acquisizione e real-time factor
(rtf = latenza / tempo tra due finestre consecutive; < 1 = tiene il passo col sonar).
La GT e' chiusa (pareti e fondo): completezza, copertura, chamfer e F-score usano solo il fondale
(top_surface), altrimenti pareti e fondo, mai visti dal sonar, peserebbero come mancanti.
L'accuratezza invece e' misurata contro la GT intera.

Percorsi e default (split, limit, metodi, n_campioni, tau) nella sezione "evaluation"
di config/config.yaml; split, limit, metodi, tau e sliding window sovrascrivibili da riga di comando.

Ogni esecuzione scrive in una cartella nuova, <evaluation.output_dir>/<data_ora>_<split>[_sw]/:
  config.txt                 comando, config effettivamente usata (con le opzioni da riga di comando),
                             macchina (CPU, GPU, versioni) e commit git
  git_diff.patch             solo se c'erano modifiche non committate: il loro diff
  log.txt                    tutto l'output del terminale, anche dei sottoprocessi
  metriche_per_terreno.csv   una riga per (terreno, metodo)
  metriche_per_famiglia.csv  medie per (famiglia, metodo)
  finestre.csv               solo sliding window: una riga per (terreno, metodo, finestra) con
                             ping, punti, latenza_s, tempo_metodo_s, acquisizione_s, budget_s, rtf
  meshes/<id>_<metodo>[_sw].ply
  distanze/<id>_<metodo>[_sw]_{rec2gt,gt2rec}.npy  distanze grezze (float32, n_campioni ciascuno):
    rec2gt = punti della ricostruzione -> fondale, gt2rec = punti del fondale -> ricostruzione;
    precisione/richiamo/F-score/Chamfer per qualsiasi tau si ricalcolano da qui.
Con sliding_window.abilitato in config (o --sliding_window on) la ricostruzione avviene a finestre di ping (suffisso _sw).

Uso:
	python evaluate.py --split val --limit 2 --methods poisson ppsurf   # 2 terreni per famiglia
	python evaluate.py --split test+val --methods poisson ppsurf --sliding_window on
	python evaluate.py --ricalcola output/evaluation/<run> --tau 0.5 0.2 0.1   # altre soglie, senza ricostruire
	python evaluate.py --methods ppsurf --set ppsurf.gen_refine_iter=5          # un parametro diverso dal config

Con --ricalcola non si ricostruisce nulla: dalle distanze grezze di una run gia' fatta si ricalcolano
Chamfer, precisione, richiamo e F-score per ogni tau indicato, e si scrivono nella cartella della run
metriche_tau.csv (una riga per terreno, metodo e tau) e metriche_tau_per_famiglia.csv (medie).
"""

import argparse
import atexit
import copy
import datetime
import glob
import os
import sys
import traceback

import numpy as np
import open3d as o3d
import pandas as pd
import yaml

from utils.mesh import top_surface
from utils.run_info import ambiente, stato_git, TeeLog, MonitorRisorse
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


def statistiche_distanze(d: np.ndarray, prefisso: str) -> dict:
	"""Mediana, scarto interquartile e 95-esimo percentile di un vettore di distanze."""
	p25, p50, p75, p95 = np.percentile(d, [25, 50, 75, 95])
	return {f"{prefisso}_mediana": float(p50), f"{prefisso}_iqr": float(p75 - p25), f"{prefisso}_p95": float(p95)}


def riassunto_finestre(finestre: list) -> dict:
	"""Colonne per terreno dalle statistiche delle finestre (sliding window)."""
	ok = [f for f in finestre if f["esito"] == "ok"]
	if not ok:
		return {}
	latenze = np.array([f["latenza_s"] for f in ok])
	rtf = np.array([f["rtf"] for f in ok if f["rtf"] is not None])
	riassunto = {"n_finestre": len(ok), "latenza_mediana_s": float(np.median(latenze)),
				 "latenza_max_s": float(latenze.max())}
	if len(rtf):
		riassunto.update(rtf_mediano=float(np.median(rtf)), rtf_max=float(rtf.max()),
						 finestre_in_tempo=float((rtf <= 1).mean()))
	return riassunto


def calcola_metriche(mesh: o3d.geometry.TriangleMesh, gt_mesh: o3d.geometry.TriangleMesh,
					 fondale: o3d.geometry.TriangleMesh, tau: float, n_campioni: int):
	"""Metriche aggregate + distanze grezze {"rec2gt": ricostruzione->fondale, "gt2rec": fondale->ricostruzione}."""
	punti_mesh = np.asarray(mesh.sample_points_uniformly(n_campioni).points)
	punti_fondale = np.asarray(fondale.sample_points_uniformly(n_campioni).points)

	acc = distanze_da_mesh(punti_mesh, gt_mesh)
	comp = distanze_da_mesh(punti_fondale, mesh)
	acc_fondale = distanze_da_mesh(punti_mesh, fondale)
	metriche = {
		"accuratezza_media": float(acc.mean()),
		"accuratezza_p90": float(np.percentile(acc, 90)),
		"completezza_media": float(comp.mean()),
		"completezza_p90": float(np.percentile(comp, 90)),
		"copertura": float((comp <= tau).mean()),
		**metriche_superficie(acc_fondale, comp, tau),
		**statistiche_distanze(acc_fondale, "rec2gt"),
		**statistiche_distanze(comp, "gt2rec"),
	}
	return metriche, {"rec2gt": acc_fondale, "gt2rec": comp}


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
CSV_FINESTRE = "finestre.csv"


def cartella_run(output_dir: str, split: str, sliding_window: bool) -> str:
	"""Cartella nuova per questa esecuzione: <output_dir>/<data_ora>_<split>[_sw]."""
	ora = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	return os.path.join(output_dir, f"{ora}_{split}{'_sw' if sliding_window else ''}")


def ultima_run(output_dir: str, split: str, sliding_window: bool = None) -> str:
	"""Cartella della run piu' recente di uno split (per gli script di benchmark che ne leggono le mesh).
	sliding_window: True/False per filtrare le run con/senza SW, None = qualsiasi."""
	suffissi = {None: ["", "_sw"], False: [""], True: ["_sw"]}[sliding_window]
	cartelle = sorted(c for s in suffissi for c in glob.glob(os.path.join(output_dir, f"*_{split}{s}")))
	if not cartelle:
		raise FileNotFoundError(f"Nessuna run di evaluate.py per lo split {split} in {output_dir}")
	return cartelle[-1]  # il nome comincia con data e ora: l'ordine alfabetico e' quello temporale


def salva_config(config: dict, cartella_out: str):
	"""config.txt: comando, config effettiva della run (YAML), macchina e versione del codice.
	Se ci sono modifiche non committate, il loro diff va in git_diff.patch."""
	python_ppsurf = None
	if "ppsurf" in config["evaluation"]["metodi"]:
		try:
			from scripts.ppsurf import trova_python_ambiente_conda
			python_ppsurf = trova_python_ambiente_conda(config["ppsurf"]["conda_env"], "")
		except Exception:
			pass
	git, diff = stato_git()
	with open(os.path.join(cartella_out, "config.txt"), "w") as f:
		f.write(f"# comando: python {' '.join(sys.argv)}\n")
		f.write(f"# avvio:   {datetime.datetime.now().isoformat(timespec='seconds')}\n")
		f.write("# config effettiva (config/config.yaml + opzioni da riga di comando):\n\n")
		yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
		f.write("\n# ---- macchina e versioni ----\n")
		yaml.safe_dump({"ambiente": ambiente(python_ppsurf)}, f, sort_keys=False, allow_unicode=True)
		f.write("\n# ---- codice ----\n")
		yaml.safe_dump({"git": git}, f, sort_keys=False, allow_unicode=True)
	if diff:
		with open(os.path.join(cartella_out, "git_diff.patch"), "w") as f:
			f.write(diff + "\n")


def salva_csv(righe: list, cartella_out: str):
	"""CSV per terreno e medie per famiglia di questa run."""
	df = pd.DataFrame(righe)
	df.to_csv(os.path.join(cartella_out, CSV_TERRENO), index=False, float_format="%.3f")

	ok = df[df["errore"].isna()] if "errore" in df else df
	if ok.empty:
		return
	colonne = ["accuratezza_media", "accuratezza_p90", "completezza_media", "completezza_p90",
			   "copertura", "chamfer", "precisione", "richiamo", "fscore",
			   "rec2gt_mediana", "rec2gt_iqr", "rec2gt_p95", "gt2rec_mediana", "gt2rec_iqr", "gt2rec_p95",
			   "n_punti_input", "n_vertici", "tempo_calcolo_secondi", "ram_picco_mb", "vram_picco_mb",
			   "latenza_mediana_s", "latenza_max_s", "rtf_mediano", "rtf_max", "finestre_in_tempo"]
	colonne = [c for c in colonne if c in ok.columns]  # le colonne della SW esistono solo con la SW
	gruppi = ["famiglia", "metodo", "sliding_window"]
	media_e_std(ok, gruppi, colonne).to_csv(
		os.path.join(cartella_out, CSV_FAMIGLIA), index=False, float_format="%.3f")


COLONNE_STD = ["chamfer", "precisione", "richiamo", "fscore"]


def media_e_std(df: pd.DataFrame, gruppi: list, colonne: list) -> pd.DataFrame:
	"""Medie per gruppo, con <col>_std (deviazione standard tra terreni) subito dopo le metriche di qualita'.
	La std e' vuota con un solo terreno nel gruppo."""
	g = df.groupby(gruppi)
	medie = g[colonne].mean()
	std = g[[c for c in COLONNE_STD if c in colonne]].std()
	ordine = []
	for c in colonne:
		ordine.append(c)
		if c in std.columns:
			medie[f"{c}_std"] = std[c]
			ordine.append(f"{c}_std")
	medie = medie[ordine]
	medie.insert(0, "n_terreni", g.size())
	return medie.reset_index()


def salva_finestre(finestre: list, cartella_out: str, id_terreno: str, metodo: str):
	"""Righe per finestra in coda a finestre.csv (una scrittura per terreno: un crash non perde i precedenti)."""
	df = pd.DataFrame(finestre)
	df.insert(0, "id", id_terreno)
	df.insert(1, "metodo", metodo)
	percorso = os.path.join(cartella_out, CSV_FINESTRE)
	df.to_csv(percorso, mode="a", header=not os.path.exists(percorso), index=False, float_format="%.3f")


def ricalcola_tau(cartella_run: str, tau: list):
	"""Metriche a soglie tau diverse dalle distanze grezze di una run (nessuna ricostruzione)."""
	per_terreno = pd.read_csv(os.path.join(cartella_run, CSV_TERRENO))
	if "errore" in per_terreno:
		per_terreno = per_terreno[per_terreno["errore"].isna()]
	righe = []
	for _, r in per_terreno.iterrows():
		suffisso = "_sw" if str(r["sliding_window"]).lower() == "true" else ""
		base = os.path.join(cartella_run, "distanze", f"{r['id']}_{r['metodo']}{suffisso}")
		if not os.path.exists(f"{base}_rec2gt.npy"):
			print(f"  distanze mancanti per {r['id']} {r['metodo']}: saltato")
			continue
		d_rec, d_gt = np.load(f"{base}_rec2gt.npy"), np.load(f"{base}_gt2rec.npy")
		for t in tau:
			righe.append({"id": r["id"], "famiglia": r["famiglia"], "metodo": r["metodo"],
						  "sliding_window": r["sliding_window"], "tau": t, **metriche_superficie(d_rec, d_gt, t)})
	if not righe:
		sys.exit(f"Nessuna distanza in {cartella_run}/distanze")
	df = pd.DataFrame(righe)
	df.to_csv(os.path.join(cartella_run, "metriche_tau.csv"), index=False, float_format="%.3f")
	gruppi = ["famiglia", "metodo", "sliding_window", "tau"]
	media_e_std(df, gruppi, COLONNE_STD).to_csv(
		os.path.join(cartella_run, "metriche_tau_per_famiglia.csv"), index=False, float_format="%.3f")
	tutti = df.groupby(["metodo", "sliding_window", "tau"])[["chamfer", "precisione", "richiamo", "fscore"]].mean()
	print(tutti.round(3).to_string())
	print(f"\nScritti metriche_tau.csv e metriche_tau_per_famiglia.csv in {cartella_run}")


def main():
	config_base = load_config()
	ev = config_base["evaluation"]

	parser = argparse.ArgumentParser(description="Valuta i metodi di ricostruzione sui terreni sintetici.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--limit", type=int, default=ev["limit"], help="numero massimo di terreni per famiglia")
	parser.add_argument("--methods", nargs="+", default=ev["metodi"], choices=list(ESECUTORI))
	parser.add_argument("--tau", type=float, nargs="+", default=[ev["tau"]],
						help="soglia in metri; con --ricalcola se ne possono dare piu' d'una")
	parser.add_argument("--ricalcola", metavar="CARTELLA_RUN", default=None,
						help="non ricostruisce: ricalcola le metriche per ogni --tau dalle distanze di questa run")
	parser.add_argument("--set", nargs="+", action="extend", default=[], metavar="SEZIONE.CHIAVE=VALORE",
						help="sovrascrive valori del config, es. --set ppsurf.gen_refine_iter=5 poisson.profondita=8")
	parser.add_argument("--sliding_window", choices=["on", "off"], default=None,
						help="sovrascrive sliding_window.abilitato del config")
	args = parser.parse_args()

	if args.ricalcola:
		ricalcola_tau(args.ricalcola, args.tau)
		return
	if len(args.tau) > 1:
		parser.error("piu' valori di --tau solo con --ricalcola")
	args.tau = args.tau[0]

	# le opzioni da riga di comando entrano nella config, cosi' config.txt descrive la run davvero eseguita
	config_base = copy.deepcopy(config_base)
	for assegnazione in args.set:
		chiave, sep, valore = assegnazione.partition("=")
		if not sep:
			parser.error(f"--set vuole SEZIONE.CHIAVE=VALORE, non '{assegnazione}'")
		*sezioni, ultima = chiave.split(".")
		nodo = config_base
		for sezione in sezioni:
			if not isinstance(nodo.get(sezione), dict):
				parser.error(f"--set: sezione '{sezione}' non presente nel config")
			nodo = nodo[sezione]
		if ultima not in nodo:
			parser.error(f"--set: chiave '{chiave}' non presente nel config")
		nodo[ultima] = yaml.safe_load(valore)  # 5 -> int, 0.5 -> float, null -> None, true -> bool
	if args.sliding_window is not None:
		config_base.setdefault("sliding_window", {})["abilitato"] = args.sliding_window == "on"
	ev = config_base["evaluation"]
	ev.update(split=args.split, limit=args.limit, metodi=args.methods, tau=args.tau)
	sliding_window = bool(config_base.get("sliding_window", {}).get("abilitato"))
	suffisso = "_sw" if sliding_window else ""

	cartella_out = cartella_run(ev["output_dir"], args.split, sliding_window)
	cartella_mesh = os.path.join(cartella_out, "meshes")
	cartella_distanze = os.path.join(cartella_out, "distanze")
	os.makedirs(cartella_mesh, exist_ok=True)
	os.makedirs(cartella_distanze, exist_ok=True)
	# tutto l'output del terminale (anche dei sottoprocessi) finisce anche in log.txt
	tee = TeeLog(os.path.join(cartella_out, "log.txt")).__enter__()
	atexit.register(tee.__exit__, None, None, None)
	print(f"Run in: {cartella_out}")
	salva_config(config_base, cartella_out)

	famiglie = famiglie_da_manifest(ev["synthetic_dir"])
	terreni = elenca_terreni(ev["synthetic_dir"], args.split, famiglie, args.limit)
	righe = []

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
			riga = {"id": id_terreno, "famiglia": famiglia, "metodo": metodo, "sliding_window": sliding_window}
			try:
				config = config_per_terreno(config_base, percorso_scan, metodo)
				finestre = []
				with MonitorRisorse() as risorse:
					mesh, pcd, tempo, nome = ricostruisci(config, finestre)
				riga["metodo"] = nome
				# punti che il metodo ha ricevuto dopo il preprocessing (con la SW: somma sulle finestre,
				# i ping in sovrapposizione contano due volte)
				riga["n_punti_input"] = (sum(f.get("n_punti_puliti") or 0 for f in finestre) if finestre
										 else len(pcd.points))
				riga["ram_picco_mb"] = risorse.ram_picco_mb
				riga["vram_picco_mb"] = risorse.vram_picco_mb

				mesh_out = os.path.join(cartella_mesh, f"{id_terreno}_{nome}{suffisso}.ply")
				o3d.io.write_triangle_mesh(mesh_out, mesh)

				metriche, distanze = calcola_metriche(mesh, gt_mesh, fondale, args.tau, ev["n_campioni"])
				riga.update(metriche)
				# distanze grezze: altre soglie tau ricalcolabili offline senza rifare la ricostruzione
				for verso, d in distanze.items():
					np.save(os.path.join(cartella_distanze, f"{id_terreno}_{nome}{suffisso}_{verso}.npy"),
							d.astype(np.float32))
				riga.update({"n_vertici": len(mesh.vertices), "tempo_calcolo_secondi": tempo,
							 "mesh": mesh_out, "errore": None})
				if finestre:
					riga.update(riassunto_finestre(finestre))
					salva_finestre(finestre, cartella_out, id_terreno, nome)
				print("  " + ", ".join(f"{k}={v:.4f}" for k, v in riga.items() if isinstance(v, float)))
			except Exception as e:
				traceback.print_exc()
				riga["errore"] = f"{type(e).__name__}: {e}"
			righe.append(riga)
			# riscritto a ogni passo: un crash non perde i risultati
			salva_csv(righe, cartella_out)

	print(f"\nRisultati in: {cartella_out}")


if __name__ == "__main__":
	main()
