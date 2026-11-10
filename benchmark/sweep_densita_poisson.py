"""
Sweep della soglia di taglio per densita' di Poisson (poisson.rimuovi_bassa_densita) sui terreni sintetici.

Per ogni terreno Poisson viene eseguito UNA volta sola (stesso preprocessing e stesse normali di
evaluate.py / utils/pipeline.py); poi la mesh grezza viene tagliata a ogni quantile, pulita e
misurata contro il fondale della GT con Chamfer, precisione, richiamo e F-score (stessa funzione
di evaluate.py). Serve a scegliere il quantile con l'F-score migliore e a vedere se il valore
migliore e' stabile tra famiglie di terreni (se no: meglio una soglia adattiva).

Output:
  benchmark/benchmark_output/sweep_densita_poisson.csv   una riga per (terreno, quantile)
  a terminale: F-score medio per quantile e quantile migliore per terreno/famiglia

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/sweep_densita_poisson.py                        # split/limit da config (sezione evaluation)
	python benchmark/sweep_densita_poisson.py --split val --limit 1  # 1 terreno per famiglia
	python benchmark/sweep_densita_poisson.py --quantili 0.02 0.05 0.1
"""

import argparse
import os
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distanze_da_mesh, metriche_superficie, elenca_terreni, famiglia_di, famiglie_da_manifest
from scripts.poisson import poisson_grezzo, taglia_bassa_densita, pulisci_mesh
from utils.mesh import stima_normali, top_surface
from utils.pipeline import load_config, config_per_terreno, carica_e_pulisci_point_cloud

QUANTILI_DEFAULT = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.35]
FILE_OUTPUT = "benchmark/benchmark_output/sweep_densita_poisson.csv"


def main():
	config_base = load_config()
	ev = config_base["evaluation"]
	p = config_base["poisson"]

	parser = argparse.ArgumentParser(description="Sweep di poisson.rimuovi_bassa_densita sui terreni sintetici.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--limit", type=int, default=ev["limit"], help="terreni per famiglia (0 = tutti)")
	parser.add_argument("--quantili", type=float, nargs="+", default=QUANTILI_DEFAULT)
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	famiglie = famiglie_da_manifest(ev["synthetic_dir"])
	terreni = elenca_terreni(ev["synthetic_dir"], args.split, famiglie, args.limit)
	righe = []

	for i, id_terreno in enumerate(terreni, 1):
		cartella = os.path.join(ev["synthetic_dir"], args.split, id_terreno)
		print(f"\n=== [{i}/{len(terreni)}] {id_terreno} ===")
		config = config_per_terreno(config_base, os.path.join(cartella, f"{id_terreno}_scan.csv"), "poisson")

		t0 = time.time()
		pcd = stima_normali(carica_e_pulisci_point_cloud(config))
		mesh_grezza, densita = poisson_grezzo(pcd, profondita=p["profondita"])
		print(f"  Poisson grezzo in {time.time() - t0:.1f}s")

		fondale = top_surface(o3d.io.read_triangle_mesh(os.path.join(cartella, f"{id_terreno}_ground_truth.obj")))
		punti_fondale = np.asarray(fondale.sample_points_uniformly(ev["n_campioni"]).points)

		for q in args.quantili:
			mesh = pulisci_mesh(taglia_bassa_densita(mesh_grezza, densita, q),
								dimensione_min_componente=p["dimensione_min_componente"])
			if len(mesh.triangles) == 0:
				print(f"  q={q}: mesh vuota dopo il taglio")
				continue
			punti_mesh = np.asarray(mesh.sample_points_uniformly(ev["n_campioni"]).points)
			m = metriche_superficie(distanze_da_mesh(punti_mesh, fondale),
									distanze_da_mesh(punti_fondale, mesh), args.tau)
			righe.append({"id": id_terreno, "famiglia": famiglia_di(id_terreno, famiglie),
						  "quantile": q, "soglia_densita": float(np.quantile(densita, q)), **m})
			print(f"  q={q:<5} chamfer={m['chamfer']:.3f}  P={m['precisione']:.3f}  "
				  f"R={m['richiamo']:.3f}  F={m['fscore']:.3f}")

	df = pd.DataFrame(righe)
	os.makedirs(os.path.dirname(FILE_OUTPUT), exist_ok=True)
	df.to_csv(FILE_OUTPUT, index=False, float_format="%.3f")
	print(f"\nRisultati per terreno e quantile in {FILE_OUTPUT}")

	pd.set_option("display.width", 200)
	print(f"\n=== Media su {df['id'].nunique()} terreni, per quantile (attuale in config: {p['rimuovi_bassa_densita']}) ===")
	print(df.groupby("quantile")[["chamfer", "precisione", "richiamo", "fscore"]].mean().round(3).to_string())

	print("\n=== F-score per famiglia e quantile ===")
	print(df.pivot_table(index="famiglia", columns="quantile", values="fscore").round(3).to_string())

	migliori = df.loc[df.groupby("id")["fscore"].idxmax(), ["id", "famiglia", "quantile", "fscore"]]
	print("\n=== Quantile migliore per terreno (se varia molto: meglio una soglia adattiva) ===")
	print(migliori.to_string(index=False))


if __name__ == "__main__":
	main()
