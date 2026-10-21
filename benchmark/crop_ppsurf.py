"""
Diagnosi griglia vs modello per PPSurf: ricostruzione su un crop di un terreno.

Con gen_resolution_global = 129 la griglia di marching cubes copre l'intera scena: su un
terreno di 100 m sono celle di ~0.8 m (Poisson a depth 9: ~0.21 m). Su un crop di 25 m, a
parita' di 129, le celle diventano ~0.2 m senza aumentare la VRAM. L'input resta lo stesso
della run completa (stesso preprocessing, voxel compreso): cambia solo la griglia.

Sullo stesso quadrato interno (crop meno un margine, per escludere gli artefatti di bordo
del crop) si misurano contro il fondale della GT, con la stessa funzione di evaluate.py:
  completo_originale / completo_medio   mesh PPSurf della run su tutto il terreno (celle ~0.8 m)
  crop_originale     / crop_medio       nuova run PPSurf sul crop (celle ~0.2 m)
  poisson                               mesh Poisson di evaluate.py, ritagliata
"medio" = punto medio tra i due strati del guscio doppio (come strato_singolo_ppsurf.py).

Se crop_medio scende verso Poisson il limite e' la griglia; se resta ~ completo_medio e' il modello.
NOTA: crop_medio scende, ma il crop riduce insieme cella ed estensione della scena; con la
griglia a 519 sull'intero terreno (stessa cella) il guscio resta ~0.9 m: conta l'estensione
(vedi README e tile_ppsurf.py).

Richiede le mesh di evaluate.py (<id>_ppsurf.ply, <id>_poisson.ply) e lancia UNA run PPSurf
(GPU, qualche minuto) tramite scripts/ppsurf.py, con i parametri della sezione ppsurf del config.

Output:
  output/benchmark/crop_ppsurf.csv
  output/benchmark/ppsurf_crop/<id>_crop<lato>_{originale,medio}.ply

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/crop_ppsurf.py                              # ridges_012, crop 25 m al centro
	python benchmark/crop_ppsurf.py --id rocky_012 --lato 25 --centro 10 -5
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.strato_singolo_ppsurf import primo_hit_verticale, mesh_da_heightmap, misura
from scripts.ppsurf import ricostruisci_ppsurf
from utils.mesh import top_surface
from utils.pipeline import load_config, config_per_terreno, carica_e_pulisci_point_cloud

FILE_OUTPUT = "output/benchmark/crop_ppsurf.csv"
CARTELLA_MESH = "output/benchmark/ppsurf_crop"
WORK_DIR = "output/benchmark/ppsurf_crop_tmp"


def quadrato(centro, lato: float) -> o3d.geometry.AxisAlignedBoundingBox:
	"""Box XZ centrato in centro, illimitato (in pratica) in Y."""
	cx, cz = centro
	return o3d.geometry.AxisAlignedBoundingBox(np.array([cx - lato / 2, -1e4, cz - lato / 2]),
											   np.array([cx + lato / 2, 1e4, cz + lato / 2]))


def medio(mesh: o3d.geometry.TriangleMesh, box: o3d.geometry.AxisAlignedBoundingBox, passo: float):
	"""Strato singolo (punto medio tra primo hit dall'alto e dal basso) su una griglia dentro box."""
	lo, hi = box.get_min_bound(), box.get_max_bound()
	X, Z = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, passo), np.arange(lo[2], hi[2] + 1e-9, passo))
	y = np.asarray(mesh.vertices)[:, 1]
	alto = primo_hit_verticale(mesh, X, Z, y.max() + 10, -1.0)
	basso = primo_hit_verticale(mesh, X, Z, y.min() - 10, 1.0)
	alto[~np.isfinite(alto)] = np.nan
	basso[~np.isfinite(basso)] = np.nan
	return mesh_da_heightmap(X, (alto + basso) / 2, Z), float(np.nanmedian(alto - basso))


def main():
	config_base = load_config()
	ev = config_base["evaluation"]
	pps = config_base["ppsurf"]

	parser = argparse.ArgumentParser(description="PPSurf su un crop: la griglia 129 e' il limite?")
	parser.add_argument("--id", default="ridges_012")
	parser.add_argument("--split", default="val", choices=["train", "val", "test"])
	parser.add_argument("--lato", type=float, default=25.0, help="lato del crop, m")
	parser.add_argument("--margine", type=float, default=2.5, help="bordo del crop escluso dalle metriche, m")
	parser.add_argument("--centro", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Z"))
	parser.add_argument("--passo", type=float, default=0.25, help="passo della griglia per lo strato medio, m")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	cartella = os.path.join(ev["synthetic_dir"], args.split, args.id)
	cartella_mesh_eval = os.path.join(ev["output_dir"], args.split, "meshes")
	percorsi = {m: os.path.join(cartella_mesh_eval, f"{args.id}_{m}.ply") for m in ("ppsurf", "poisson")}
	mancanti = [p for p in percorsi.values() if not os.path.exists(p)]
	if mancanti:
		raise SystemExit(f"Mesh mancanti (lancia prima evaluate.py): {mancanti}")

	box_crop = quadrato(args.centro, args.lato)
	box_interno = quadrato(args.centro, args.lato - 2 * args.margine)
	nome_run = f"{args.id}_crop{args.lato:g}"

	# --- input: stesso preprocessing della run completa, poi solo i punti del crop
	config = config_per_terreno(config_base, os.path.join(cartella, f"{args.id}_scan.csv"), "ppsurf")
	pcd = carica_e_pulisci_point_cloud(config).crop(box_crop)
	print(f"\n[crop] {args.lato:g}x{args.lato:g} m attorno a {args.centro}: {len(pcd.points)} punti "
		  f"(celle marching cubes ~{args.lato * 1.0 / (pps['gen_resolution_global'] - 1):.2f} m)")

	mesh_crop, tempo = ricostruisci_ppsurf(
		pcd, nome_run,
		ppsurf_dir=os.path.join(config_base["paths"]["neural_methods"], "ppsurf"),
		conda_env=pps["conda_env"], work_dir=WORK_DIR,
		gen_resolution_global=pps["gen_resolution_global"], rec_batch_size=pps["rec_batch_size"],
	)

	# --- tutte le varianti misurate sullo stesso quadrato interno
	fondale = top_surface(o3d.io.read_triangle_mesh(os.path.join(cartella, f"{args.id}_ground_truth.obj")))
	fondale = fondale.crop(box_interno)
	punti_fondale = np.asarray(fondale.sample_points_uniformly(ev["n_campioni"]).points)

	completo = o3d.io.read_triangle_mesh(percorsi["ppsurf"])
	completo_medio, spessore_completo = medio(completo, box_interno, args.passo)
	crop_medio, spessore_crop = medio(mesh_crop, box_interno, args.passo)

	os.makedirs(CARTELLA_MESH, exist_ok=True)
	o3d.io.write_triangle_mesh(os.path.join(CARTELLA_MESH, f"{nome_run}_originale.ply"), mesh_crop)
	o3d.io.write_triangle_mesh(os.path.join(CARTELLA_MESH, f"{nome_run}_medio.ply"), crop_medio)

	varianti = {
		"completo_originale": (completo.crop(box_interno), spessore_completo),
		"completo_medio": (completo_medio, spessore_completo),
		"crop_originale": (mesh_crop.crop(box_interno), spessore_crop),
		"crop_medio": (crop_medio, spessore_crop),
		"poisson": (o3d.io.read_triangle_mesh(percorsi["poisson"]).crop(box_interno), float("nan")),
	}
	righe = []
	print(f"\n=== {args.id}: metriche sul quadrato interno {args.lato - 2 * args.margine:g} m (tau={args.tau} m) ===")
	for nome, (mesh, spessore) in varianti.items():
		m = misura(mesh, fondale, punti_fondale, ev["n_campioni"], args.tau)
		righe.append({"id": args.id, "variante": nome, "lato_crop": args.lato, "n_punti_crop": len(pcd.points),
					  "spessore_mediano": spessore, "tempo_ppsurf_crop": tempo, **m})
		print(f"  {nome:19s} chamfer={m['chamfer']:.3f}  P={m['precisione']:.3f}  R={m['richiamo']:.3f}  "
			  f"F={m['fscore']:.3f}  spessore guscio={spessore:.2f} m")

	pd.DataFrame(righe).to_csv(FILE_OUTPUT, index=False, float_format="%.3f")
	print(f"\nPPSurf sul crop: {tempo:.1f}s. Risultati in {FILE_OUTPUT}, mesh in {CARTELLA_MESH}/")


if __name__ == "__main__":
	main()
