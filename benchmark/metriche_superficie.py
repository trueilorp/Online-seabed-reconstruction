"""
Metriche superficie-superficie della mesh ricostruita contro il fondale della ground truth,
usate dagli script benchmark/tempo_*.sh.

Si campionano n punti uniformi su entrambe le superfici e si misura la distanza esatta
punto-superficie (stesse funzioni di evaluate.py, quindi stessi numeri):
  d_rec = distanza dei punti della ricostruzione dal fondale GT    (-> precisione)
  d_gt  = distanza dei punti del fondale GT dalla ricostruzione    (-> richiamo)
  chamfer    = (media(d_rec) + media(d_gt)) / 2, in metri (Chamfer L1 simmetrica)
  precisione = frazione di d_rec <= tau
  richiamo   = frazione di d_gt  <= tau
  fscore     = media armonica di precisione e richiamo
Il fondale e' top_surface(GT): pareti e fondo dei terreni sintetici, mai visti dal sonar,
non devono contare come superficie mancante.

Stampa su stdout una riga CSV "chamfer,precisione,richiamo,fscore,tau" (il resto su stderr).

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/metriche_superficie.py <mesh> <gt.obj>
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import distanze_da_mesh, metriche_superficie
from utils.mesh import top_surface
from utils.pipeline import load_config


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("mesh")
	parser.add_argument("gt")
	args = parser.parse_args()

	ev = load_config()["evaluation"]
	tau, n_campioni = ev["tau"], ev["n_campioni"]

	mesh = o3d.io.read_triangle_mesh(args.mesh)
	fondale = top_surface(o3d.io.read_triangle_mesh(args.gt))

	punti_rec = np.asarray(mesh.sample_points_uniformly(n_campioni).points)
	punti_gt = np.asarray(fondale.sample_points_uniformly(n_campioni).points)
	d_rec = distanze_da_mesh(punti_rec, fondale)
	d_gt = distanze_da_mesh(punti_gt, mesh)

	m = metriche_superficie(d_rec, d_gt, tau)

	print(f"  chamfer={m['chamfer']:.4f} m, precisione={m['precisione']:.4f}, richiamo={m['richiamo']:.4f}, "
		  f"fscore={m['fscore']:.4f} (tau={tau} m)", file=sys.stderr)
	print(f"{m['chamfer']:.4f},{m['precisione']:.4f},{m['richiamo']:.4f},{m['fscore']:.4f},{tau}")


if __name__ == "__main__":
	main()
