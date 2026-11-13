"""
Grafico a barre del Chamfer per famiglia di fondale (media ± std tra i terreni), per le quattro
configurazioni Poisson / PPSurf, con e senza SW. Scala logaritmica: i due metodi differiscono di
un ordine di grandezza.

Legge metriche_per_famiglia.csv dalle quattro cartelle di run di evaluate.py.

Uso (dalla root del progetto):
	python visualization/plot_chamfer.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--out imgs/chamfer_per_fondale.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

BLU, ARANCIO = "#2a78d6", "#eb6834"      # Poisson, PPSurf
BLU_CHIARO, ARANCIO_CHIARO = "#9ec5f4", "#f6b59a"
INCHIOSTRO, INCHIOSTRO_2, GRIGLIA = "#1f1f1e", "#5f5e58", "#e4e3de"

# (argomento, etichetta, riempimento, bordo, tratteggio)
SERIE = [
	("poisson", "Poisson", BLU, BLU, None),
	("poisson_sw", "Poisson + SW", BLU_CHIARO, BLU, "////"),
	("ppsurf", "PPSurf", ARANCIO, ARANCIO, None),
	("ppsurf_sw", "PPSurf + SW", ARANCIO_CHIARO, ARANCIO, "////"),
]


def leggi(cartella_run: str) -> pd.DataFrame:
	df = pd.read_csv(os.path.join(cartella_run, "metriche_per_famiglia.csv"))
	return df.set_index("famiglia")[["chamfer", "chamfer_std"]]


def main():
	parser = argparse.ArgumentParser(description="Chamfer per famiglia, 4 configurazioni")
	for nome, *_ in SERIE:
		parser.add_argument(f"--{nome}", required=True, help="cartella di run")
	parser.add_argument("--out", default="imgs/chamfer_per_fondale.png")
	args = parser.parse_args()

	dati = {nome: leggi(getattr(args, nome)) for nome, *_ in SERIE}
	famiglie = list(dati["poisson"].index)
	x = np.arange(len(famiglie))
	larghezza, spazio = 0.19, 0.02

	plt.rcParams.update({"font.size": 10, "hatch.linewidth": 0.8})
	fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
	for i, (nome, etichetta, colore, bordo, tratteggio) in enumerate(SERIE):
		d = dati[nome].reindex(famiglie)
		pos = x + (i - 1.5) * (larghezza + spazio)
		plt.rcParams["hatch.color"] = bordo
		ax.bar(pos, d["chamfer"], larghezza, label=etichetta, color=colore, edgecolor=bordo,
			   linewidth=0.8, hatch=tratteggio, zorder=2)
		ax.errorbar(pos, d["chamfer"], yerr=d["chamfer_std"].fillna(0), fmt="none",
					ecolor=INCHIOSTRO, elinewidth=0.8, capsize=2, zorder=3)

	ax.set_yscale("log")
	ax.set_ylim(0.02, 1)
	tacche = [0.03, 0.05, 0.1, 0.2, 0.5, 1]
	ax.yaxis.set_major_locator(FixedLocator(tacche))
	ax.yaxis.set_minor_locator(NullLocator())
	ax.set_yticklabels([f"{t:g}" for t in tacche])
	ax.set_ylabel("Chamfer [m] (più basso è meglio)", color=INCHIOSTRO_2)
	ax.set_xticks(x, famiglie)
	ax.grid(axis="y", color=GRIGLIA, linewidth=0.8, zorder=0)
	for lato in ("top", "right"):
		ax.spines[lato].set_visible(False)
	for lato in ("left", "bottom"):
		ax.spines[lato].set_color(INCHIOSTRO_2)
	ax.tick_params(colors=INCHIOSTRO_2)
	ax.legend(ncols=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Salvato {args.out}")


if __name__ == "__main__":
	main()
