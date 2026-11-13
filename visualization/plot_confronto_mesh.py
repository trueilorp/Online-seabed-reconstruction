"""
Griglia di screenshot: per ogni terreno scelto, ground truth e le quattro configurazioni
(Poisson, Poisson + SW, PPSurf, PPSurf + SW) affiancate, con la stessa inquadratura.

Usa gli screenshot gia' salvati da visualization/screenshot_run.py:
  <run>/screenshots/<id>_<metodo>[_sw].png   e   <cartella_gt>/<id>_ground_truth.png
I bordi bianchi comuni a tutte le immagini di una riga vengono tagliati.

Uso (dalla root del progetto):
	python visualization/plot_confronto_mesh.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--terreni flat_sand_012 rocky_012 mixed_012 --out imgs/confronto_mesh.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# (argomento o None per la GT, titolo colonna, suffisso del file)
COLONNE = [
	(None, "Ground truth", "_ground_truth"),
	("poisson", "Poisson", "_poisson"),
	("poisson_sw", "Poisson + SW", "_poisson_sw"),
	("ppsurf", "PPSurf", "_ppsurf"),
	("ppsurf_sw", "PPSurf + SW", "_ppsurf_sw"),
]


def riquadro_non_bianco(immagini: list, soglia: float = 0.98):
	"""Bounding box comune dei pixel non bianchi di una lista di immagini (stesse dimensioni)."""
	maschera = np.zeros(immagini[0].shape[:2], dtype=bool)
	for im in immagini:
		maschera |= im[..., :3].min(axis=2) < soglia
	righe, colonne = np.where(maschera)
	return righe.min(), righe.max() + 1, colonne.min(), colonne.max() + 1


def main():
	parser = argparse.ArgumentParser(description="Griglia GT + 4 configurazioni per alcuni terreni")
	for nome, *_ in COLONNE[1:]:
		parser.add_argument(f"--{nome}", required=True, help="cartella di run con screenshots/")
	parser.add_argument("--ground_truth", default="output/groundtruth", help="cartella degli screenshot della GT")
	parser.add_argument("--terreni", nargs="+", default=["flat_sand_012", "rocky_012", "mixed_012"])
	parser.add_argument("--out", default="imgs/confronto_mesh.png")
	args = parser.parse_args()

	fig, assi = plt.subplots(len(args.terreni), len(COLONNE), figsize=(13, 2.3 * len(args.terreni)), dpi=150,
							 squeeze=False)
	for r, terreno in enumerate(args.terreni):
		immagini = []
		for nome, _, suffisso in COLONNE:
			cartella = args.ground_truth if nome is None else os.path.join(getattr(args, nome), "screenshots")
			immagini.append(mpimg.imread(os.path.join(cartella, f"{terreno}{suffisso}.png")))
		y0, y1, x0, x1 = riquadro_non_bianco(immagini)
		for c, (im, (_, titolo, _)) in enumerate(zip(immagini, COLONNE)):
			ax = assi[r, c]
			ax.imshow(im[y0:y1, x0:x1])
			ax.set_xticks([])
			ax.set_yticks([])
			for lato in ax.spines.values():
				lato.set_visible(False)
			if r == 0:
				ax.set_title(titolo, fontsize=11)
			if c == 0:
				ax.set_ylabel(terreno, fontsize=10)
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Salvato {args.out}")


if __name__ == "__main__":
	main()
