"""
Preparazione dell'input per gli script di benchmark dei tempi (benchmark/tempo_*.sh).

Esegue il preprocessing comune della pipeline (caricamento CSV + pulizia, come
utils/pipeline.py) UNA volta sola, salva la point cloud in <out_dir>/input.ply e
scrive <out_dir>/params.sh con i parametri del metodo letti da config/config.yaml,
cosi' gli script .sh possono cronometrare i comandi dei metodi isolati dal resto.

La sliding window viene ignorata: si misura la ricostruzione sull'intera point cloud.

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/prepara_input.py --metodo ppsurf --out_dir benchmark/benchmark_output/ppsurf/<dataset> [--csv file.csv]
"""

import argparse
import os
import shlex
import sys
import time

import open3d as o3d
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import load_config, carica_e_pulisci_point_cloud, nome_run_da_config
from scripts.ppsurf import sottocampiona_se_necessario


def salva_solo_punti(pcd: o3d.geometry.PointCloud, percorso: str):
	os.makedirs(os.path.dirname(percorso), exist_ok=True)
	pcd_solo_punti = o3d.geometry.PointCloud()
	pcd_solo_punti.points = pcd.points
	o3d.io.write_point_cloud(percorso, pcd_solo_punti)


def percorso_ground_truth(config: dict, csv: str) -> str:
	"""GT per le metriche: <id>_ground_truth.obj accanto a <id>_scan.csv (terreni sintetici),
	altrimenti paths.ground_truth_mesh se il CSV e' quello del config. "" se non c'e'."""
	if csv:
		cartella, nome = os.path.split(csv)
		id_terreno = os.path.splitext(nome)[0].removesuffix("_scan")
		candidato = os.path.join(cartella, f"{id_terreno}_ground_truth.obj")
	else:
		candidato = config["paths"].get("ground_truth_mesh") or ""
	if candidato and os.path.isfile(candidato):
		return os.path.abspath(candidato)
	print("[prepara_input] Nessuna ground truth trovata: metriche di superficie non calcolate")
	return ""


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--metodo", required=True, choices=["poisson", "ppsurf", "sa_convonet"])
	parser.add_argument("--out_dir", required=True)
	parser.add_argument("--csv", default=None, help="CSV di input (default: paths.input_csv del config)")
	args = parser.parse_args()

	config = load_config()
	if args.csv:
		config["paths"]["input_csv"] = args.csv
	config["paths"].pop("nome_run", None)
	nome_run = nome_run_da_config(config)
	out_dir = os.path.abspath(args.out_dir)
	os.makedirs(out_dir, exist_ok=True)

	inizio = time.time()
	pcd = carica_e_pulisci_point_cloud(config)
	print(f"[prepara_input] Caricamento + pulizia: {time.time() - inizio:.2f}s, {len(pcd.points)} punti")

	params = {"NOME_RUN": nome_run, "GT_MESH": percorso_ground_truth(config, args.csv)}

	if args.metodo == "ppsurf":
		pps = config["ppsurf"]
		pcd = sottocampiona_se_necessario(pcd, pps["max_punti"])
		params.update(
			CONDA_ENV=pps["conda_env"],
			METODO_DIR=os.path.abspath(os.path.join(config["paths"]["neural_methods"], "ppsurf")),
			PPS_RES=pps["gen_resolution_global"],
			PPS_BATCH=pps["rec_batch_size"],
		)
		salva_solo_punti(pcd, os.path.join(out_dir, "input.ply"))

	elif args.metodo == "sa_convonet":
		sac = config["sa_convonet"]
		# stessa struttura di cartelle e stesso run_config di scripts/sa_convonet.py (senza ottimizzazione online)
		salva_solo_punti(pcd, os.path.join(out_dir, "raw", nome_run, "pointcloud.ply"))
		run_config = {
			"inherit_from": sac["config_file"],
			"data": {
				"path": os.path.join(out_dir, "normalize"),
				"unit_size": sac["unit_size"],
				"query_vol_size": sac["query_vol_size"],
			},
			"training": {"out_dir": os.path.join(out_dir, "out")},
		}
		with open(os.path.join(out_dir, "run_config.yaml"), "w") as f:
			yaml.safe_dump(run_config, f)
		params.update(
			CONDA_ENV=sac["conda_env"],
			METODO_DIR=os.path.abspath(os.path.join(config["paths"]["neural_methods"], "SA-ConvONet")),
		)

	else:
		salva_solo_punti(pcd, os.path.join(out_dir, "input.ply"))

	params["N_PUNTI"] = len(pcd.points)
	with open(os.path.join(out_dir, "params.sh"), "w") as f:
		for chiave, valore in params.items():
			f.write(f"{chiave}={shlex.quote(str(valore))}\n")
	print(f"[prepara_input] Input pronto in {out_dir} ({params['N_PUNTI']} punti per {args.metodo})")


if __name__ == "__main__":
	main()
