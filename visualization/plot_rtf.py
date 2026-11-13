"""
Real-time factor per finestra (sliding window): Poisson + SW contro PPSurf + SW.

Legge finestre.csv (una riga per terreno, metodo e finestra, scritto da evaluate.py con
--sliding_window on) dalle due cartelle di run. Asse x = indice della finestra lungo la rotta,
asse y = rtf = latenza_s / budget_s in scala logaritmica. Linea = mediana tra i terreni,
fascia = 10-90 percentile tra i terreni. La linea orizzontale a RTF = 1 separa le finestre
elaborate in tempo (sotto) da quelle che si accumulano (sopra).

Uso (dalla root del progetto):
	python visualization/plot_rtf.py \
		--poisson_sw output/evaluation/<run_poisson_sw> \
		--ppsurf_sw  output/evaluation/<run_ppsurf_sw> \
		--out imgs/rtf_finestre.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

BLU, ARANCIO = "#2a78d6", "#eb6834"      # Poisson, PPSurf
INCHIOSTRO, INCHIOSTRO_2, GRIGLIA = "#1f1f1e", "#5f5e58", "#e4e3de"

SERIE = [("poisson_sw", "Poisson + SW", BLU), ("ppsurf_sw", "PPSurf + SW", ARANCIO)]


def leggi(cartella_run: str) -> pd.DataFrame:
	"""Mediana e percentili 10/90 di rtf tra i terreni, per indice di finestra."""
	df = pd.read_csv(os.path.join(cartella_run, "finestre.csv"))
	df = df[(df["esito"] == "ok") & df["rtf"].notna()]
	g = df.groupby("finestra")["rtf"]
	return pd.DataFrame({"mediana": g.median(), "p10": g.quantile(0.1), "p90": g.quantile(0.9),
						 "n": g.size()})


def main():
	parser = argparse.ArgumentParser(description="RTF per finestra, Poisson + SW contro PPSurf + SW")
	for nome, *_ in SERIE:
		parser.add_argument(f"--{nome}", required=True, help="cartella di run con finestre.csv")
	parser.add_argument("--out", default="imgs/rtf_finestre.png")
	args = parser.parse_args()

	plt.rcParams.update({"font.size": 10})
	fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
	for nome, etichetta, colore in SERIE:
		d = leggi(getattr(args, nome))
		ax.fill_between(d.index, d["p10"], d["p90"], color=colore, alpha=0.15, linewidth=0, zorder=1)
		ax.plot(d.index, d["mediana"], color=colore, linewidth=2, marker="o", markersize=4,
				label=etichetta, zorder=2)
	ax.axhline(1.0, color=INCHIOSTRO, linewidth=1, linestyle="--", zorder=3)
	ax.text(ax.get_xlim()[1], 1.0, " RTF = 1", va="center", ha="left", color=INCHIOSTRO, fontsize=9)

	ax.set_yscale("log")
	tacche = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
	ax.yaxis.set_major_locator(FixedLocator(tacche))
	ax.yaxis.set_minor_locator(NullLocator())
	ax.set_yticklabels([f"{t:g}" for t in tacche])
	ax.set_xlabel("finestra (ordine di acquisizione)", color=INCHIOSTRO_2)
	ax.set_ylabel("RTF = latenza / budget", color=INCHIOSTRO_2)
	ax.grid(color=GRIGLIA, linewidth=0.8, zorder=0)
	for lato in ("top", "right"):
		ax.spines[lato].set_visible(False)
	for lato in ("left", "bottom"):
		ax.spines[lato].set_color(INCHIOSTRO_2)
	ax.tick_params(colors=INCHIOSTRO_2)
	ax.legend(ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
	fig.tight_layout()

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Salvato {args.out}")


if __name__ == "__main__":
	main()
