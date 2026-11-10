"""
PPSurf a porzioni: il terreno viene diviso in quadrati XZ, ricostruiti separatamente e riuniti.

PPSurf normalizza ogni point cloud nel cubo unitario, e lo spessore del guscio doppio risulta
~1% del lato della scena qualunque sia la griglia di marching cubes (terreno di 100 m: ~0.9 m
sia con gen_resolution_global 129 sia con 519; crop di 25 m: ~0.3 m). Ricostruire quadrati
piccoli assottiglia quindi il guscio; lo strato medio (come in strato_singolo_ppsurf.py) toglie
il resto.

Quadrati: nuclei di lato (lato - 2*margine) che coprono l'impronta della scansione senza
sovrapporsi; PPSurf riceve i punti del nucleo allargato di margine per lato, cosi' i bordi di
ogni ricostruzione (artefatti) cadono fuori dal nucleo e vengono scartati nell'unione.
Tutti i quadrati di un terreno passano in UN solo sottoprocesso PPSurf (modalita' dataset di
pps.py: testset.txt + 04_pts_vis/), quindi modello e CUDA vengono caricati una volta sola.
Il tempo di PPSurf e' fissato dalla griglia, non dai punti: ogni quadrato costa quanto un
terreno intero alla stessa gen_resolution_global.

Unione e varianti misurate contro il fondale della GT (stessa funzione di evaluate.py):
  tile_originale      mesh di ogni quadrato ritagliata sul proprio nucleo, concatenate
  tile_medio          strato medio su una griglia XZ globale (passo --passo): ogni nodo prende
                      il valore dal quadrato del cui nucleo fa parte -> una sola heightmap
  completo_originale  mesh PPSurf intera di evaluate.py (se presente; attenzione: e' quella
  completo_medio      dell'ultima run di evaluate.py, con la sua gen_resolution_global)
  poisson             mesh Poisson di evaluate.py (se presente)

Lo strato medio e' 2.5D: va bene per il fondale, ma cancella sporgenze e vuoti (tubi, piloni).

Output:
  benchmark/benchmark_output/tile_ppsurf.csv                         in append, una riga per (run, terreno, variante)
  benchmark/benchmark_output/ppsurf_tile/<id>_tile<lato>_{originale,medio}.ply

Uso (dalla root del progetto, ambiente "ricostruzione3d"; parametri PPSurf dalla sezione ppsurf del config):
	python benchmark/tile_ppsurf.py                              # split da config, 1 terreno per famiglia, quadrati da 25 m
	python benchmark/tile_ppsurf.py --id rocky_007 --lato 25 --margine 2.5
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.strato_singolo_ppsurf import primo_hit_verticale, mesh_da_heightmap, misura
from evaluate import elenca_terreni, famiglia_di, famiglie_da_manifest
from scripts.ppsurf import trova_python_ambiente_conda
from utils.mesh import top_surface
from utils.pipeline import load_config, config_per_terreno, carica_e_pulisci_point_cloud

FILE_OUTPUT = "benchmark/benchmark_output/tile_ppsurf.csv"
CARTELLA_MESH = "benchmark/benchmark_output/ppsurf_tile"
WORK_DIR = "benchmark/benchmark_output/ppsurf_tile_tmp"


def box_xz(x0: float, x1: float, z0: float, z1: float) -> o3d.geometry.AxisAlignedBoundingBox:
	"""Box XZ illimitato (in pratica) in Y."""
	return o3d.geometry.AxisAlignedBoundingBox(np.array([x0, -1e4, z0]), np.array([x1, 1e4, z1]))


def strati(mesh: o3d.geometry.TriangleMesh, X: np.ndarray, Z: np.ndarray):
	"""Quota del primo hit dall'alto e dal basso nei nodi (X, Z); NaN dove il raggio non colpisce."""
	y = np.asarray(mesh.vertices)[:, 1]
	alto = primo_hit_verticale(mesh, X, Z, y.max() + 10, -1.0)
	basso = primo_hit_verticale(mesh, X, Z, y.min() - 10, 1.0)
	alto[~np.isfinite(alto)] = np.nan
	basso[~np.isfinite(basso)] = np.nan
	return alto, basso


def ricostruisci_quadrati(nuvole: dict, nome_run: str, ppsurf_dir: str, pps: dict):
	"""Tutte le nuvole {nome: pcd} in un solo sottoprocesso PPSurf (modalita' dataset di pps.py).
	Ritorna ({nome: mesh}, tempo_secondi); i quadrati senza mesh in uscita mancano dal dizionario."""
	python_ppsurf = trova_python_ambiente_conda(pps["conda_env"], ppsurf_dir)
	cartella = os.path.abspath(os.path.join(WORK_DIR, nome_run))
	shutil.rmtree(cartella, ignore_errors=True)
	cartella_dataset = os.path.join(cartella, "dataset")
	cartella_punti = os.path.join(cartella_dataset, "04_pts_vis")  # struttura attesa da pps.py
	cartella_output = os.path.join(cartella, "results")
	os.makedirs(cartella_punti)

	# in modalita' dataset pps.py assume input gia' normalizzati e non riporta le mesh in metri:
	# stessa normalizzazione che applica al file singolo (source/base/math.py,
	# get_points_normalization_info con padding_factor 0.05 di configs/poco.yaml), invertita all'uscita
	normalizzazione = {}
	for nome, pcd in nuvole.items():
		p = np.asarray(pcd.points)
		centro = (p.min(axis=0) + p.max(axis=0)) / 2
		scala = float((p.max(axis=0) - p.min(axis=0)).max() * 1.05)
		normalizzazione[nome] = (centro, scala)
		solo_punti = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((p - centro) / scala))  # niente normali
		o3d.io.write_point_cloud(os.path.join(cartella_punti, f"{nome}.xyz.ply"), solo_punti)
	testset = os.path.join(cartella_dataset, "testset.txt")
	with open(testset, "w") as f:
		f.write("\n".join(nuvole) + "\n")

	cmd = [python_ppsurf, "pps.py", "rec", testset, cartella_output,
		   "--model.init_args.gen_resolution_global", str(pps["gen_resolution_global"]),
		   "--model.init_args.rec_batch_size", str(pps["rec_batch_size"])]
	env = os.environ.copy()
	env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # vedi scripts/ppsurf.py

	print(f"\n[PPSurf] {len(nuvole)} quadrati in un solo sottoprocesso (ambiente '{pps['conda_env']}')...")
	inizio = time.time()
	risultato = subprocess.run(cmd, cwd=ppsurf_dir, env=env, capture_output=True, text=True)
	tempo = time.time() - inizio
	if risultato.returncode != 0:
		print("--- ultime righe di stdout ---\n" + risultato.stdout[-3000:])
		print("--- ultime righe di stderr ---\n" + risultato.stderr[-3000:])
		raise RuntimeError(f"PPSurf (sottoprocesso) terminato con codice {risultato.returncode}")
	print(f"[PPSurf] completato in {tempo:.1f}s ({tempo / len(nuvole):.1f}s per quadrato)")

	# modalita' dataset: <out>/<nome modello>/<cartella del dataset>/meshes/<nome>.xyz.ply
	mesh = {}
	for nome in nuvole:
		candidati = glob.glob(os.path.join(cartella_output, "*", "dataset", "meshes", f"{nome}.xyz.ply"))
		if candidati:
			m = o3d.io.read_triangle_mesh(candidati[0])
			centro, scala = normalizzazione[nome]
			m.vertices = o3d.utility.Vector3dVector(np.asarray(m.vertices) * scala + centro)
			mesh[nome] = m
		else:
			print(f"  ATTENZIONE: nessuna mesh in uscita per {nome}")
	return mesh, tempo


def main():
	config_base = load_config()
	ev = config_base["evaluation"]
	pps = config_base["ppsurf"]

	parser = argparse.ArgumentParser(description="PPSurf a porzioni (quadrati XZ) + strato medio.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test"])
	parser.add_argument("--limit", type=int, default=1, help="terreni per famiglia (0 = tutti)")
	parser.add_argument("--id", nargs="+", default=None, help="terreni specifici (ignora --limit)")
	parser.add_argument("--lato", type=float, default=25.0, help="lato del quadrato dato a PPSurf, m")
	parser.add_argument("--margine", type=float, default=2.5, help="sovrapposizione per lato, scartata nell'unione, m")
	parser.add_argument("--min-punti", type=int, default=200, help="quadrati con meno punti nel nucleo vengono saltati")
	parser.add_argument("--passo", type=float, default=0.25, help="passo della griglia XZ dello strato medio, m")
	parser.add_argument("--gen-res", type=int, default=pps["gen_resolution_global"],
						help="gen_resolution_global per quadrato (default: config)")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()
	pps = {**pps, "gen_resolution_global": args.gen_res}

	nucleo = args.lato - 2 * args.margine
	if nucleo <= 0:
		raise SystemExit("--lato deve essere maggiore di 2 * --margine")

	famiglie = famiglie_da_manifest(ev["synthetic_dir"])
	terreni = args.id or elenca_terreni(ev["synthetic_dir"], args.split, famiglie, args.limit)
	cartella_mesh_eval = os.path.join(ev["output_dir"], args.split, "meshes")
	ppsurf_dir = os.path.join(config_base["paths"]["neural_methods"], "ppsurf")
	os.makedirs(CARTELLA_MESH, exist_ok=True)
	run = datetime.datetime.now().isoformat(timespec="seconds")
	righe = []

	for i, id_terreno in enumerate(terreni, 1):
		cartella = os.path.join(ev["synthetic_dir"], args.split, id_terreno)
		nome_run = f"{id_terreno}_tile{args.lato:g}"
		print(f"\n=== [{i}/{len(terreni)}] {id_terreno} ===")

		# --- input: stesso preprocessing di evaluate.py, poi suddivisione in quadrati
		config = config_per_terreno(config_base, os.path.join(cartella, f"{id_terreno}_scan.csv"), "ppsurf")
		pcd = carica_e_pulisci_point_cloud(config)
		punti = np.asarray(pcd.points)
		lo, hi = punti.min(axis=0), punti.max(axis=0)
		nx = max(1, int(np.ceil((hi[0] - lo[0]) / nucleo)))
		nz = max(1, int(np.ceil((hi[2] - lo[2]) / nucleo)))

		nuclei, nuvole = {}, {}
		for ix in range(nx):
			for iz in range(nz):
				x0, z0 = lo[0] + ix * nucleo, lo[2] + iz * nucleo
				box_nucleo = box_xz(x0, x0 + nucleo, z0, z0 + nucleo)
				if len(box_nucleo.get_point_indices_within_bounding_box(pcd.points)) < args.min_punti:
					continue
				nome = f"{id_terreno}_x{ix:02d}_z{iz:02d}"
				nuclei[nome] = (ix, iz, box_nucleo)
				nuvole[nome] = pcd.crop(box_xz(x0 - args.margine, x0 + nucleo + args.margine,
											   z0 - args.margine, z0 + nucleo + args.margine))
		print(f"[quadrati] {len(nuvole)} su {nx}x{nz} (lato {args.lato:g} m, nucleo {nucleo:g} m, "
			  f"celle marching cubes ~{args.lato / (pps['gen_resolution_global'] - 1):.2f} m)")

		mesh_quadrati, tempo = ricostruisci_quadrati(nuvole, nome_run, ppsurf_dir, pps)

		# --- unione: originale = mesh ritagliate sui nuclei; medio = una heightmap sulla griglia globale
		X, Z = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, args.passo), np.arange(lo[2], hi[2] + 1e-9, args.passo))
		ix_nodo = np.minimum(((X - lo[0]) // nucleo).astype(int), nx - 1)
		iz_nodo = np.minimum(((Z - lo[2]) // nucleo).astype(int), nz - 1)
		alto, basso = np.full(X.shape, np.nan), np.full(X.shape, np.nan)
		tile_originale = o3d.geometry.TriangleMesh()
		for nome, mesh in mesh_quadrati.items():
			ix, iz, box_nucleo = nuclei[nome]
			tile_originale += mesh.crop(box_nucleo)
			sel = (ix_nodo == ix) & (iz_nodo == iz)
			alto[sel], basso[sel] = strati(mesh, X[sel], Z[sel])
		tile_originale.compute_vertex_normals()
		tile_medio = mesh_da_heightmap(X, (alto + basso) / 2, Z)
		o3d.io.write_triangle_mesh(os.path.join(CARTELLA_MESH, f"{nome_run}_originale.ply"), tile_originale)
		o3d.io.write_triangle_mesh(os.path.join(CARTELLA_MESH, f"{nome_run}_medio.ply"), tile_medio)
		spessore_tile = float(np.nanmedian(alto - basso))

		varianti = {"tile_originale": (tile_originale, spessore_tile), "tile_medio": (tile_medio, spessore_tile)}
		percorso_completo = os.path.join(cartella_mesh_eval, f"{id_terreno}_ppsurf.ply")
		if os.path.exists(percorso_completo):
			completo = o3d.io.read_triangle_mesh(percorso_completo)
			alto_c, basso_c = strati(completo, X, Z)
			spessore_completo = float(np.nanmedian(alto_c - basso_c))
			varianti["completo_originale"] = (completo, spessore_completo)
			varianti["completo_medio"] = (mesh_da_heightmap(X, (alto_c + basso_c) / 2, Z), spessore_completo)
		percorso_poisson = os.path.join(cartella_mesh_eval, f"{id_terreno}_poisson.ply")
		if os.path.exists(percorso_poisson):
			varianti["poisson"] = (o3d.io.read_triangle_mesh(percorso_poisson), float("nan"))

		# --- metriche contro il fondale della GT
		fondale = top_surface(o3d.io.read_triangle_mesh(os.path.join(cartella, f"{id_terreno}_ground_truth.obj")))
		punti_fondale = np.asarray(fondale.sample_points_uniformly(ev["n_campioni"]).points)
		for variante, (mesh, spessore) in varianti.items():
			m = misura(mesh, fondale, punti_fondale, ev["n_campioni"], args.tau)
			righe.append({"run": run, "id": id_terreno, "famiglia": famiglia_di(id_terreno, famiglie),
						  "variante": variante, "lato": args.lato, "margine": args.margine,
						  "gen_resolution_global": pps["gen_resolution_global"], "n_quadrati": len(mesh_quadrati),
						  "tempo_ppsurf_secondi": tempo, "spessore_mediano": spessore, **m})
			print(f"  {variante:19s} chamfer={m['chamfer']:.3f}  P={m['precisione']:.3f}  R={m['richiamo']:.3f}  "
				  f"F={m['fscore']:.3f}  spessore guscio={spessore:.2f} m")

	df = pd.DataFrame(righe)
	precedenti = pd.read_csv(FILE_OUTPUT) if os.path.exists(FILE_OUTPUT) else pd.DataFrame()
	pd.concat([precedenti, df], ignore_index=True).to_csv(FILE_OUTPUT, index=False, float_format="%.3f")
	print(f"\nRisultati in {FILE_OUTPUT} (run {run}), mesh in {CARTELLA_MESH}/")

	ordine = ["completo_originale", "completo_medio", "tile_originale", "tile_medio", "poisson"]
	media = df.groupby("variante")[["chamfer", "precisione", "richiamo", "fscore", "spessore_mediano"]].mean()
	print(f"\n=== Media su {df['id'].nunique()} terreni (tau={args.tau} m) ===")
	print(media.reindex([v for v in ordine if v in media.index]).round(3).to_string())


if __name__ == "__main__":
	main()
