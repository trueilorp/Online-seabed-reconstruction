"""
Post-processing "strato singolo" delle mesh PPSurf: toglie il guscio doppio e misura quanto
PPSurf si avvicina a Poisson senza di esso.

PPSurf (pre-addestrato su oggetti chiusi) avvolge il fondale aperto in un solido sottile
(~1 m): quasi ogni raggio verticale attraversa due superfici, una sopra e una sotto il fondale
vero. Il fondale e' 2.5D, quindi su una griglia XZ regolare si prendono:
  alto   primo hit di un raggio dall'alto (strato superiore)
  basso  primo hit di un raggio dal basso (strato inferiore)
  medio  punto medio tra i due
dove c'e' un solo strato i due coincidono, dove non c'e' nulla resta un buco. Ogni variante
torna una mesh a griglia e viene misurata contro il fondale della GT con la stessa funzione
di evaluate.py, accanto alla mesh PPSurf originale e a Poisson sullo stesso terreno.

Se "medio" scende verso Poisson, il guscio e' centrato sul fondale e il fine-tuning con
occupancy "sotto il fondale" ha buone probabilita'; se resta alto, il limite e' altrove
(griglia 129 di marching cubes, modello).

Lavora sulle mesh gia' salvate da evaluate.py (non rilancia PPSurf, niente GPU) e non le modifica.

Output:
  benchmark/benchmark_output/strato_singolo_ppsurf.csv              una riga per (terreno, variante)
  benchmark/benchmark_output/ppsurf_strato_singolo/<id>_{alto,basso,medio}.ply

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/strato_singolo_ppsurf.py                    # split da config, 1 terreno per famiglia
	python benchmark/strato_singolo_ppsurf.py --split val --limit 2 --passo 0.25
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distanze_da_mesh, metriche_superficie, elenca_terreni, famiglia_di, famiglie_da_manifest, ultima_run
from utils.mesh import top_surface
from utils.pipeline import load_config

FILE_OUTPUT = "benchmark/benchmark_output/strato_singolo_ppsurf.csv"
CARTELLA_MESH = "benchmark/benchmark_output/ppsurf_strato_singolo"


def primo_hit_verticale(mesh: o3d.geometry.TriangleMesh, X: np.ndarray, Z: np.ndarray,
						y_partenza: float, direzione: float) -> np.ndarray:
	"""Quota Y del primo hit di raggi verticali da y_partenza (direzione -1 = verso il basso); NaN se mancano."""
	scena = o3d.t.geometry.RaycastingScene()
	scena.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
	n = X.size
	raggi = np.stack([X.ravel(), np.full(n, y_partenza), Z.ravel(),
					  np.zeros(n), np.full(n, direzione), np.zeros(n)], axis=1).astype(np.float32)
	t = scena.cast_rays(o3d.core.Tensor(raggi))["t_hit"].numpy()
	return (y_partenza + direzione * t).reshape(X.shape)  # t = inf -> +-inf, poi scartato


def mesh_da_heightmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> o3d.geometry.TriangleMesh:
	"""Griglia triangolata; i triangoli con un vertice mancante (Y non finita) vengono scartati."""
	nr, nc = Y.shape
	idx = np.arange(nr * nc).reshape(nr, nc)
	a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
	c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
	tri = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])  # normali verso +Y
	valido = np.isfinite(Y.ravel())
	tri = tri[valido[tri].all(axis=1)]
	vertici = np.stack([X.ravel(), np.where(valido, Y.ravel(), 0.0), Z.ravel()], axis=1)
	mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertici), o3d.utility.Vector3iVector(tri))
	mesh.remove_unreferenced_vertices()
	mesh.compute_vertex_normals()
	return mesh


def misura(mesh: o3d.geometry.TriangleMesh, fondale: o3d.geometry.TriangleMesh,
		   punti_fondale: np.ndarray, n_campioni: int, tau: float) -> dict:
	punti_mesh = np.asarray(mesh.sample_points_uniformly(n_campioni).points)
	return metriche_superficie(distanze_da_mesh(punti_mesh, fondale),
							   distanze_da_mesh(punti_fondale, mesh), tau)


def main():
	config = load_config()
	ev = config["evaluation"]

	parser = argparse.ArgumentParser(description="Strato singolo dalle mesh PPSurf (guscio doppio) e metriche.")
	parser.add_argument("--split", default=ev["split"], choices=["train", "val", "test", "test+val"])
	parser.add_argument("--run_dir", default=None,
						help="cartella di una run di evaluate.py senza SW (default: la piu' recente dello split)")
	parser.add_argument("--limit", type=int, default=1, help="terreni per famiglia (0 = tutti)")
	parser.add_argument("--passo", type=float, default=0.25, help="passo della griglia XZ, m")
	parser.add_argument("--tau", type=float, default=ev["tau"])
	args = parser.parse_args()

	cartella_mesh_eval = os.path.join(args.run_dir or ultima_run(ev["output_dir"], args.split, False), "meshes")
	famiglie = famiglie_da_manifest(ev["synthetic_dir"])
	terreni = [t for t in elenca_terreni(ev["synthetic_dir"], args.split, famiglie, args.limit)
			   if os.path.exists(os.path.join(cartella_mesh_eval, f"{t}_ppsurf.ply"))]
	if not terreni:
		raise SystemExit(f"Nessuna mesh PPSurf in {cartella_mesh_eval}: lancia prima evaluate.py --methods ppsurf")
	os.makedirs(CARTELLA_MESH, exist_ok=True)
	righe = []

	for i, id_terreno in enumerate(terreni, 1):
		print(f"\n=== [{i}/{len(terreni)}] {id_terreno} ===")
		gt = o3d.io.read_triangle_mesh(os.path.join(ev["synthetic_dir"], args.split, id_terreno,
													f"{id_terreno}_ground_truth.obj"))
		fondale = top_surface(gt)
		punti_fondale = np.asarray(fondale.sample_points_uniformly(ev["n_campioni"]).points)
		ppsurf = o3d.io.read_triangle_mesh(os.path.join(cartella_mesh_eval, f"{id_terreno}_ppsurf.ply"))

		# griglia sull'impronta comune di PPSurf e GT (fuori dalla GT non c'e' fondale da misurare)
		v_pps, v_gt = np.asarray(ppsurf.vertices), np.asarray(gt.vertices)
		lo = np.maximum(v_pps.min(axis=0), v_gt.min(axis=0))
		hi = np.minimum(v_pps.max(axis=0), v_gt.max(axis=0))
		X, Z = np.meshgrid(np.arange(lo[0], hi[0], args.passo), np.arange(lo[2], hi[2], args.passo))
		y_sopra, y_sotto = v_pps[:, 1].max() + 10, v_pps[:, 1].min() - 10

		alto = primo_hit_verticale(ppsurf, X, Z, y_sopra, -1.0)
		basso = primo_hit_verticale(ppsurf, X, Z, y_sotto, 1.0)
		alto[~np.isfinite(alto)] = np.nan
		basso[~np.isfinite(basso)] = np.nan
		spessore = alto - basso
		print(f"  griglia {X.shape[1]}x{X.shape[0]} (passo {args.passo} m): raggi con hit {np.isfinite(alto).mean():.1%}, "
			  f"spessore mediano {np.nanmedian(spessore):.2f} m")

		varianti = {"alto": alto, "basso": basso, "medio": (alto + basso) / 2}
		mesh_da_misurare = {"ppsurf_originale": ppsurf}
		for nome, Y in varianti.items():
			mesh = mesh_da_heightmap(X, Y, Z)
			o3d.io.write_triangle_mesh(os.path.join(CARTELLA_MESH, f"{id_terreno}_{nome}.ply"), mesh)
			mesh_da_misurare[f"ppsurf_{nome}"] = mesh
		percorso_poisson = os.path.join(cartella_mesh_eval, f"{id_terreno}_poisson.ply")
		if os.path.exists(percorso_poisson):
			mesh_da_misurare["poisson"] = o3d.io.read_triangle_mesh(percorso_poisson)

		for variante, mesh in mesh_da_misurare.items():
			m = misura(mesh, fondale, punti_fondale, ev["n_campioni"], args.tau)
			righe.append({"id": id_terreno, "famiglia": famiglia_di(id_terreno, famiglie),
						  "variante": variante, "spessore_mediano": float(np.nanmedian(spessore)), **m})
			print(f"  {variante:17s} chamfer={m['chamfer']:.3f}  P={m['precisione']:.3f}  "
				  f"R={m['richiamo']:.3f}  F={m['fscore']:.3f}")

	df = pd.DataFrame(righe)
	df.to_csv(FILE_OUTPUT, index=False, float_format="%.3f")
	print(f"\nRisultati in {FILE_OUTPUT}, mesh in {CARTELLA_MESH}/")

	ordine = ["ppsurf_originale", "ppsurf_alto", "ppsurf_basso", "ppsurf_medio", "poisson"]
	media = df.groupby("variante")[["chamfer", "precisione", "richiamo", "fscore"]].mean()
	print(f"\n=== Media su {df['id'].nunique()} terreni (tau={args.tau} m) ===")
	print(media.reindex([v for v in ordine if v in media.index]).round(3).to_string())
	print("\n=== Chamfer per terreno ===")
	tabella = df.pivot_table(index="id", columns="variante", values="chamfer")
	print(tabella[[v for v in ordine if v in tabella.columns]].round(3).to_string())


if __name__ == "__main__":
	main()
