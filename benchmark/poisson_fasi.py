"""
Poisson cronometrato fase per fase (stima normali, ricostruzione, pulizia mesh),
con gli stessi parametri di utils/pipeline.py:esegui_poisson. Usato da
benchmark/tempo_poisson.sh: ogni fase viene aggiunta come riga al CSV dei tempi.

Salva anche, fuori dal cronometro e accanto a <mesh_output.ply>, la diagnostica per
scegliere poisson.rimuovi_bassa_densita (come nel tutorial Open3D, che colora la mesh
per densita' prima di tagliare):
  mesh_poisson_densita.ply   mesh PRIMA del taglio, colorata per densita' (colormap plasma:
                             scuro = bassa densita', giallo = alta)
  istogramma_densita.png     distribuzione delle densita' per vertice, con la soglia attuale
Se la superficie spuria forma una "gobba" separata di valori bassi, la soglia giusta e'
nella valle tra le due.

Uso (dalla root del progetto, ambiente "ricostruzione3d"):
	python benchmark/poisson_fasi.py <input.ply> <tempi.csv> <dataset> <ripetizione> <mesh_output.ply>
"""

import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mesh import stima_normali
from utils.pipeline import load_config
from scripts.poisson import poisson_grezzo, taglia_bassa_densita, pulisci_mesh


def salva_diagnostica_densita(mesh: o3d.geometry.TriangleMesh, densita: np.ndarray,
							  quantile: float, cartella: str, titolo: str):
	norm = (densita - densita.min()) / (densita.max() - densita.min() + 1e-12)
	colorata = o3d.geometry.TriangleMesh(mesh)
	colorata.vertex_colors = o3d.utility.Vector3dVector(plt.get_cmap("plasma")(norm)[:, :3])
	colorata.compute_vertex_normals()
	o3d.io.write_triangle_mesh(os.path.join(cartella, "mesh_poisson_densita.ply"), colorata)

	soglia = np.quantile(densita, quantile)
	fig, ax = plt.subplots(figsize=(8, 4.5))
	ax.hist(densita, bins=150, color="#4c72b0")
	ax.set_yscale("log")  # una gobba piccola di superficie spuria resta visibile accanto al fondale
	ax.axvline(soglia, color="#c44e52", linestyle="--",
			   label=f"soglia attuale: quantile {quantile} = {soglia:.2f}")
	for q in (0.05, 0.1, 0.2):
		ax.axvline(np.quantile(densita, q), color="grey", linewidth=0.8, alpha=0.6)
	ax.set_xlabel("densita' Poisson per vertice")
	ax.set_ylabel("n. vertici (scala log)")
	ax.set_title(f"{titolo} - linee grigie: quantili 0.05 / 0.1 / 0.2")
	ax.legend()
	fig.tight_layout()
	fig.savefig(os.path.join(cartella, "istogramma_densita.png"), dpi=120)
	plt.close(fig)

	quantili = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9]
	print("  densita' per quantile: " + ", ".join(f"q{q}={np.quantile(densita, q):.2f}" for q in quantili))
	print(f"  diagnostica densita' salvata in {cartella}")


def main():
	percorso_ply, file_tempi, dataset, ripetizione, mesh_output = sys.argv[1:6]
	p = load_config()["poisson"]
	pcd = o3d.io.read_point_cloud(percorso_ply)

	fasi = []
	t0 = time.time()
	pcd = stima_normali(pcd)
	fasi.append(("normali", time.time() - t0))

	t0 = time.time()
	mesh_grezza, densita = poisson_grezzo(pcd, profondita=p["profondita"])
	tempo_grezzo = time.time() - t0
	salva_diagnostica_densita(mesh_grezza, densita, p["rimuovi_bassa_densita"],
							  os.path.dirname(os.path.abspath(mesh_output)), dataset)
	t0 = time.time()
	mesh = taglia_bassa_densita(mesh_grezza, densita, p["rimuovi_bassa_densita"])
	fasi.append(("ricostruzione", tempo_grezzo + time.time() - t0))

	t0 = time.time()
	mesh = pulisci_mesh(mesh, dimensione_min_componente=p["dimensione_min_componente"])
	fasi.append(("pulizia_mesh", time.time() - t0))
	o3d.io.write_triangle_mesh(mesh_output, mesh)

	# metriche di superficie lasciate vuote: tempo_poisson.sh le aggiunge sulla riga processo_completo
	data = datetime.now().astimezone().isoformat(timespec="seconds")  # stesso formato di "date -Iseconds"
	with open(file_tempi, "a") as f:
		for fase, secondi in fasi:
			print(f"  >> poisson / {fase}: {secondi:.2f}s")
			f.write(f"{data},poisson,{dataset},{len(pcd.points)},{ripetizione},{fase},{secondi:.2f},,,,,,,\n")


if __name__ == "__main__":
	main()
