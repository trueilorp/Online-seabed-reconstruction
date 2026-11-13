"""
Precisione, richiamo e F-score in funzione di tau per le quattro configurazioni Poisson / PPSurf,
con e senza SW. Linea = media tra le famiglie, fascia = minimo-massimo tra le famiglie.

Legge metriche_tau_per_famiglia.csv (scritto da evaluate.py --ricalcola) dalle quattro cartelle
di run: piu' valori di tau si ricalcolano, piu' la curva e' fine, es.
	python evaluate.py --ricalcola <run> --tau 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.75 1.0

Uso (dalla root del progetto):
	python visualization/plot_tau.py \
		--poisson output/evaluation/<run> --poisson_sw output/evaluation/<run_sw> \
		--ppsurf output/evaluation/<run> --ppsurf_sw output/evaluation/<run_sw> \
		--out imgs/metriche_tau.png
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLU, ARANCIO = "#2a78d6", "#eb6834"      # Poisson, PPSurf
INCHIOSTRO_2, GRIGLIA = "#5f5e58", "#e4e3de"

# (argomento, etichetta, colore, stile linea, marcatore)
SERIE = [
	("poisson", "Poisson", BLU, "-", "o"),
	("poisson_sw", "Poisson + SW", BLU, "--", "s"),
	("ppsurf", "PPSurf", ARANCIO, "-", "o"),
	("ppsurf_sw", "PPSurf + SW", ARANCIO, "--", "s"),
]
METRICHE = [("precisione", "Precision"), ("richiamo", "Recall"), ("fscore", "F-score")]


def leggi(cartella_run: str) -> pd.DataFrame:
	"""Per ogni tau: media, minimo e massimo tra le famiglie di ciascuna metrica."""
	df = pd.read_csv(os.path.join(cartella_run, "metriche_tau_per_famiglia.csv"))
	colonne = [m for m, _ in METRICHE]
	return df.groupby("tau")[colonne].agg(["mean", "min", "max"])


def main():
	parser = argparse.ArgumentParser(description="Precisione, richiamo e F-score in funzione di tau")
	for nome, *_ in SERIE:
		parser.add_argument(f"--{nome}", required=True, help="cartella di run")
	parser.add_argument("--out", default="imgs/metriche_tau.png")
	args = parser.parse_args()

	dati = {nome: leggi(getattr(args, nome)) for nome, *_ in SERIE}

	plt.rcParams.update({"font.size": 10})
	fig, assi = plt.subplots(1, 3, figsize=(11, 3.8), dpi=150, sharey=True)
	for ax, (metrica, titolo) in zip(assi, METRICHE):
		for nome, etichetta, colore, stile, marcatore in SERIE:
			d = dati[nome][metrica]
			tau = d.index.values
			ax.fill_between(tau, d["min"], d["max"], color=colore, alpha=0.12, linewidth=0, zorder=1)
			ax.plot(tau, d["mean"], stile, color=colore, marker=marcatore, markersize=5,
					linewidth=2, label=etichetta, zorder=2)
		ax.set_title(titolo, fontsize=11)
		ax.set_xlabel("τ [m]", color=INCHIOSTRO_2)
		ax.set_ylim(0, 1.02)
		ax.grid(color=GRIGLIA, linewidth=0.8, zorder=0)
		for lato in ("top", "right"):
			ax.spines[lato].set_visible(False)
		for lato in ("left", "bottom"):
			ax.spines[lato].set_color(INCHIOSTRO_2)
		ax.tick_params(colors=INCHIOSTRO_2)
	for ax in assi:
		ax.set_xticks(sorted(dati["poisson"].index.values))
	gestori, etichette = assi[0].get_legend_handles_labels()
	fig.legend(gestori, etichette, ncols=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
	fig.tight_layout(rect=(0, 0, 1, 0.92))

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	fig.savefig(args.out, facecolor="white")
	print(f"Salvato {args.out}")


if __name__ == "__main__":
	main()
