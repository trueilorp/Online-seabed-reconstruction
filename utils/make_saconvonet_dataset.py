"""
Converte i terreni sintetici (data/synthetic) nel dataset di training di SA-ConvONet
(input_type pointcloud_crop), per il fine-tuning con neural_methods/SA-ConvONet/fine_tuning.py.

Per ogni terreno <split>/<id>/ scrive <out>/seabed/<id>/:
  - pointcloud.npz : la SCANSIONE simulata (<id>_scan.csv), non la GT: e' il dominio da
                     imparare. Preprocessata come in inferenza (utils/pipeline.py: voxel,
                     outlier) e normalizzata come make_cropscene_dataset2.py
                     (loc = centro del bounding box, scale = estensione massima, float16),
                     cosi' unit_size corrisponde agli stessi metri in training e in inferenza.
  - points_iou.npz : punti query con occupancy vera, nello stesso spazio normalizzato.
                     Occupato = sotto il fondale, cioe' un raggio verso +Y colpisce la
                     top_surface della GT. Non si usa il dentro/fuori della GT chiusa:
                     la sua base (1 m sotto il terreno) diventerebbe una superficie da imparare.
e le liste <out>/seabed/<split>.lst. Lo split test non viene convertito di default:
resta riservato alla valutazione finale con evaluate.py.

Uso (dalla root del progetto, ambiente ricostruzione3d):
	python utils/make_saconvonet_dataset.py
	python utils/make_saconvonet_dataset.py --splits train val --limit 2
"""

import argparse
import os
import sys
import traceback

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # root del progetto

from utils.mesh import top_surface
from utils.pipeline import load_config, config_per_terreno, carica_e_pulisci_point_cloud

CATEGORIA = "seabed"  # deve coincidere con data.classes nel config di fine-tuning


def normalizzazione_scena(punti: np.ndarray):
	"""Stessa normalizzazione di make_cropscene_dataset2.py --do_norm (usata in inferenza)."""
	vert_max, vert_min = punti.max(axis=0), punti.min(axis=0)
	return (vert_min + vert_max) / 2, (vert_max - vert_min).max()


def campiona_superficie(mesh: o3d.geometry.TriangleMesh, n: int, rng: np.random.Generator) -> np.ndarray:
	"""Punti uniformi per area sulla mesh (con rng esplicito, quindi riproducibili)."""
	v = np.asarray(mesh.vertices)
	t = np.asarray(mesh.triangles)
	a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
	aree = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
	idx = rng.choice(len(t), size=n, p=aree / aree.sum())
	u, w = rng.random(n), rng.random(n)
	fuori = u + w > 1
	u[fuori], w[fuori] = 1 - u[fuori], 1 - w[fuori]
	return a[idx] + u[:, None] * (b[idx] - a[idx]) + w[:, None] * (c[idx] - a[idx])


def occupato_sotto_fondale(punti: np.ndarray, fondale: o3d.geometry.TriangleMesh) -> np.ndarray:
	"""True se il punto e' sotto il fondale: un raggio verso +Y incontra la superficie."""
	scena = o3d.t.geometry.RaycastingScene()
	scena.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(fondale))
	raggi = np.concatenate([punti, np.tile([0.0, 1.0, 0.0], (len(punti), 1))], axis=1)
	t_hit = scena.cast_rays(o3d.core.Tensor(raggi.astype(np.float32)))["t_hit"].numpy()
	return np.isfinite(t_hit)


def converti_terreno(config: dict, percorso_gt: str, cartella_out: str, n_punti: int,
					 frazione_superficie: float, rng: np.random.Generator) -> dict:
	sac = config["sa_convonet"]
	voxel = sac["unit_size"]                              # lato voxel, unita' normalizzate
	meta_crop = sac["query_vol_size"] * voxel / 2         # mezzo crop, unita' normalizzate

	# --- input: scansione con lo stesso preprocessing dell'inferenza (esegui_sa_convonet)
	pcd = carica_e_pulisci_point_cloud(config)
	punti = np.asarray(pcd.points)
	loc, scale = normalizzazione_scena(punti)
	# normali fittizie come make_cropscene_dataset2.py senza nx,ny,nz: la rete non le usa,
	# ma il dataloader (PatchPointCloudField) si aspetta la chiave
	normali = np.full_like(punti, 1 / np.sqrt(3))
	np.savez(os.path.join(cartella_out, "pointcloud.npz"),
			 points=((punti - loc) / scale).astype(np.float16),
			 normals=normali.astype(np.float16),
			 loc=loc.astype(np.float64), scale=np.float64(scale))

	# --- query: vicino al fondale (rumore di 1 e 3 voxel) + uniformi nella zona raggiungibile dai crop
	fondale = top_surface(o3d.io.read_triangle_mesh(percorso_gt))
	n_sup = int(n_punti * frazione_superficie)
	sigma = np.repeat([voxel, 3 * voxel], [n_sup // 2, n_sup - n_sup // 2]) * scale  # in metri
	vicini = campiona_superficie(fondale, n_sup, rng) + rng.normal(size=(n_sup, 3)) * sigma[:, None]

	# i centri dei crop sono uniformi nel bounding box dell'input: si allarga di mezzo crop,
	# ma in X/Z si resta dentro l'impronta della GT (fuori non c'e' terreno da etichettare)
	lb, ub = punti.min(axis=0) - meta_crop * scale, punti.max(axis=0) + meta_crop * scale
	gt_min, gt_max = np.asarray(fondale.vertices).min(axis=0), np.asarray(fondale.vertices).max(axis=0)
	for asse in (0, 2):
		lb[asse], ub[asse] = max(lb[asse], gt_min[asse]), min(ub[asse], gt_max[asse])
	uniformi = lb + rng.random((n_punti - n_sup, 3)) * (ub - lb)

	query = np.concatenate([vicini, uniformi])[rng.permutation(n_punti)]
	# l'occupancy si calcola sulle coordinate gia' arrotondate a float16 (quelle che vedra' la rete)
	query_norm = ((query - loc) / scale).astype(np.float16)
	occ = occupato_sotto_fondale(query_norm.astype(np.float64) * scale + loc, fondale)
	np.savez(os.path.join(cartella_out, "points_iou.npz"),
			 points=query_norm, occupancies=np.packbits(occ))

	return {"punti_input": len(punti), "scale_m": float(scale), "voxel_m": float(voxel * scale),
			"frazione_occupata": float(occ.mean())}


def main():
	config_base = load_config()
	ev = config_base["evaluation"]
	default_out = os.path.join(config_base["paths"]["neural_methods"], "SA-ConvONet", "data", "seabed_dataset")

	parser = argparse.ArgumentParser(description="Terreni sintetici -> dataset SA-ConvONet per il fine-tuning.")
	parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
	parser.add_argument("--limit", type=int, default=None, help="numero massimo di terreni per split")
	parser.add_argument("--out", default=default_out)
	parser.add_argument("--n-punti", type=int, default=500_000, help="punti query per terreno")
	parser.add_argument("--frazione-superficie", type=float, default=0.8,
						help="frazione di punti query vicino al fondale (il resto uniforme)")
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	cartella_categoria = os.path.join(args.out, CATEGORIA)
	os.makedirs(cartella_categoria, exist_ok=True)
	rng = np.random.default_rng(args.seed)

	for split in args.splits:
		cartella_split = os.path.join(ev["synthetic_dir"], split)
		ids = sorted(d for d in os.listdir(cartella_split) if os.path.isdir(os.path.join(cartella_split, d)))
		ids = ids[:args.limit] if args.limit else ids
		convertiti = []
		for i, id_terreno in enumerate(ids, 1):
			print(f"\n=== [{split} {i}/{len(ids)}] {id_terreno} ===")
			cartella = os.path.join(cartella_split, id_terreno)
			try:
				config = config_per_terreno(config_base, os.path.join(cartella, f"{id_terreno}_scan.csv"), "sa_convonet")
				cartella_out = os.path.join(cartella_categoria, id_terreno)
				os.makedirs(cartella_out, exist_ok=True)
				info = converti_terreno(config, os.path.join(cartella, f"{id_terreno}_ground_truth.obj"),
										cartella_out, args.n_punti, args.frazione_superficie, rng)
				print("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()))
				convertiti.append(id_terreno)
			except Exception:
				traceback.print_exc()
		with open(os.path.join(cartella_categoria, f"{split}.lst"), "w") as f:
			f.write("\n".join(convertiti))
		print(f"\n{split}: {len(convertiti)}/{len(ids)} terreni convertiti")

	print(f"\nDataset in: {args.out}")


if __name__ == "__main__":
	main()
