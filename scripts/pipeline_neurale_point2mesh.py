"""
Pipeline neurale di ricostruzione 3D da point cloud (sonar) con Point2Mesh
============================================================================

Point2Mesh (https://github.com/ranahanocka/point2mesh) ottimizza i pesi di
una CNN per deformare una mesh iniziale (qui: un convex hull) fino a farla
aderire alla point cloud di input ("self-prior", per-shape, nessun training
su un dataset esterno).

Nota architetturale importante: questo script gira nell'ambiente conda
"ricostruzione3d" (Open3D 0.19, Python 3.10) per riusare le funzioni di
mesh_builder.py. Point2Mesh richiede però un ambiente separato "point2mesh"
(PyTorch 1.5.1 + PyTorch3D 0.2.5, Python 3.8), incompatibile nello stesso
processo. Per questo il cuore dell'ottimizzazione (repo_esterni/point2mesh/main.py)
viene lanciato come sottoprocesso, usando l'interprete Python di quell'ambiente.

Dipendenze:
    ambiente "ricostruzione3d": open3d, numpy, pandas (vedi mesh_builder.py)
    ambiente "point2mesh": vedi repo_esterni/point2mesh/environment.yml
    software "Manifold" compilato in repo_esterni/Manifold/build (manifold, simplify)

Uso:
    conda activate ricostruzione3d
    cd scripts
    python pipeline_neurale_point2mesh.py
"""

import json
import os
import re
import subprocess
import time

import numpy as np
import open3d as o3d

from mesh_builder import (
    carica_point_cloud,
    pulisci_point_cloud,
    stima_normali,
    valuta_mesh,
    stampa_metriche,
    salva_metriche_csv,
)

# --- Imposta qui i parametri, stesso stile di mesh_builder.py ---
INPUT_CSV = "../dati/sonar/OS3D_Max_Range_Test.csv"  # percorso al CSV da ricostruire
ESCLUDI_CLASSI = [2]                        # valori di 'classification' da escludere
ALTITUDE_MIN = None                         # tieni solo punti con altitude >= questo valore (None = nessun limite)
ALTITUDE_MAX = None                         # tieni solo punti con altitude <= questo valore (None = nessun limite)
NB_NEIGHBORS = 20                           # filtro outlier: quanti vicini considerare per punto
STD_RATIO = 2.0                             # filtro outlier: soglia in deviazioni standard
ESCLUDI_ZONA_X = (-420, -380)               # tuple (min, max) per escludere una zona rettangolare X (sosta barca)
ESCLUDI_ZONA_Z = (5590, 5650)               # tuple (min, max) per escludere una zona rettangolare Z (sosta barca)
VOXEL_SIZE = 0.5                            # dimensione voxel per downsampling in metri
OUTPUT_MESH = "../output/mesh/mesh_point2mesh.ply"  # percorso file mesh di output
METRICHE_CSV = "../output/metriche_ricostruzione.csv"  # file dove accumulare le metriche di ogni run
CALCOLA_AUTOINTERSEZIONI = False            # True = controllo completo (lento su mesh grandi)
MAX_PUNTI_FEDELTA = 50_000                  # numero massimo di punti usati per il calcolo di fedeltà (raycasting)

# --- Parametri specifici di Point2Mesh ---
MAX_PUNTI_POINT2MESH = 8_000                # subsample aggressivo dopo pulizia: GPU 4GB VRAM + ottimizzazione
                                             # per-shape (lenta, non un forward pass) rendono proibitivo l'uso
                                             # delle centinaia di migliaia di punti del dataset sonar completo
ITERATIONS = 1500                           # iterazioni di ottimizzazione. Con --upsamp di default (1000,
                                             # non esposto qui, vedi options.py di Point2Mesh) un solo upsampling
                                             # scatta a iter 999 (2000 -> 3000 facce), lasciando 500 iterazioni
                                             # per allenare la mesh piu' fine. NON usare 2000: l'upsampling
                                             # scatterebbe di nuovo esattamente all'ultima iterazione, esportando
                                             # una mesh appena rimescolata ma mai allenata (verificato: peggiora
                                             # la fedelta' rispetto a un run piu' corto)
EXPORT_INTERVAL = 100                       # ogni quante iterazioni Point2Mesh esporta una mesh intermedia
INITIAL_MESH_FACES = 2000                   # numero di facce target per la mesh iniziale (convex hull semplificato)
MANIFOLD_RES = 20000                        # risoluzione dell'octree usata dal software Manifold per il remesh

REPO_ESTERNI = "../repo_esterni"
POINT2MESH_DIR = os.path.join(REPO_ESTERNI, "point2mesh")
MANIFOLD_DIR = os.path.join(REPO_ESTERNI, "Manifold", "build")
POINT2MESH_CONDA_ENV = "point2mesh"         # nome dell'ambiente conda dedicato a Point2Mesh
WORK_DIR = "../output/point2mesh_tmp"       # point cloud/mesh iniziale temporanee scambiate col sottoprocesso


# ---------------------------------------------------------------------------
# UTILITA'
# ---------------------------------------------------------------------------

def trova_python_ambiente_conda(nome_ambiente: str) -> str:
    """Risolve il path dell'interprete python di un ambiente conda dato il nome,
    senza richiedere che sia attivo nella shell corrente."""
    risultato = subprocess.run(
        ["conda", "env", "list", "--json"], capture_output=True, text=True, check=True
    )
    ambienti = json.loads(risultato.stdout)["envs"]
    for percorso in ambienti:
        if os.path.basename(percorso.rstrip("/")) == nome_ambiente:
            python_bin = os.path.join(percorso, "bin", "python")
            if not os.path.isfile(python_bin):
                raise FileNotFoundError(f"Interprete python non trovato in {python_bin}")
            return python_bin
    raise RuntimeError(
        f"Ambiente conda '{nome_ambiente}' non trovato. Crealo con: "
        f"conda env create -f {POINT2MESH_DIR}/environment.yml"
    )


def subsample_aggressivo(pcd: o3d.geometry.PointCloud, max_punti: int) -> o3d.geometry.PointCloud:
    """Riduce ulteriormente la point cloud sotto una soglia fissa. Necessario per
    Point2Mesh: a differenza di Poisson/BPA (istantanei), qui ogni punto pesa sul
    costo di ogni iterazione di ottimizzazione, e la GPU ha solo 4GB di VRAM."""
    n = len(pcd.points)
    if n <= max_punti:
        print(f"  Punti ({n}) gia' sotto la soglia Point2Mesh ({max_punti}), nessun ulteriore subsample")
        return pcd
    frazione = max_punti / n
    pcd_sub = pcd.random_down_sample(frazione)
    print(f"  Subsample aggressivo per Point2Mesh: {n} -> {len(pcd_sub.points)} punti "
          f"(GPU 4GB VRAM, ottimizzazione per-shape)")
    return pcd_sub


def esporta_point_cloud_point2mesh(pcd: o3d.geometry.PointCloud, percorso_output_ply: str):
    """Point2Mesh legge un file .ply ASCII con ESATTAMENTE 6 valori per riga
    (x y z nx ny nz) tramite un parser testuale minimale (utils.read_pts nel
    repo Point2Mesh) che non conosce lo schema PLY: prende i primi 3 valori
    come xyz e TUTTO il resto della riga come normale. Se la point cloud ha
    anche colori (usati da mesh_builder per colorare in base a intensita'/
    altezza), Open3D li scriverebbe come 3 colonne aggiuntive, facendo leggere
    a Point2Mesh una "normale" a 6 componenti invece di 3 -> va quindi esportata
    una copia senza colori, indipendentemente da pcd."""
    if not pcd.has_normals():
        raise ValueError("La point cloud non ha normali stimate: chiama stima_normali() prima")
    os.makedirs(os.path.dirname(percorso_output_ply), exist_ok=True)
    pcd_senza_colori = o3d.geometry.PointCloud()
    pcd_senza_colori.points = pcd.points
    pcd_senza_colori.normals = pcd.normals
    o3d.io.write_point_cloud(percorso_output_ply, pcd_senza_colori, write_ascii=True)
    print(f"  Point cloud esportata per Point2Mesh: {percorso_output_ply} ({len(pcd.points)} punti)")


def genera_mesh_iniziale_convex_hull(pcd: o3d.geometry.PointCloud, percorso_output_obj: str,
                                      num_facce_target: int = 2000, manifold_res: int = 20000):
    """Genera la mesh iniziale richiesta da Point2Mesh: un convex hull della point
    cloud (valido per forme di genus 0, come da README del progetto), rimesciato
    e semplificato al numero di facce target con il software Manifold -- stesso
    procedimento di scripts/process_data/convex_hull.py del repo Point2Mesh, ma
    implementato qui con Open3D per evitare la dipendenza da trimesh."""
    manifold_bin = os.path.join(MANIFOLD_DIR, "manifold")
    simplify_bin = os.path.join(MANIFOLD_DIR, "simplify")
    if not (os.path.isfile(manifold_bin) and os.path.isfile(simplify_bin)):
        raise FileNotFoundError(
            f"Binari Manifold non trovati in {MANIFOLD_DIR}. Compilali prima (cmake + make)."
        )

    os.makedirs(os.path.dirname(percorso_output_obj), exist_ok=True)
    hull, _ = pcd.compute_convex_hull()
    hull.compute_vertex_normals()
    n_facce_hull = len(hull.triangles)
    print(f"  Convex hull grezzo: {len(hull.vertices)} vertici, {n_facce_hull} facce")

    percorso_hull_grezzo = percorso_output_obj.replace(".obj", "_hull_grezzo.obj")
    percorso_hull_remeshato = percorso_output_obj.replace(".obj", "_hull_remesh.obj")
    o3d.io.write_triangle_mesh(percorso_hull_grezzo, hull)

    cmd_manifold = [manifold_bin, percorso_hull_grezzo, percorso_hull_remeshato, str(manifold_res)]
    ris = subprocess.run(cmd_manifold, capture_output=True, text=True)
    if ris.returncode != 0 or not os.path.isfile(percorso_hull_remeshato):
        raise RuntimeError(f"Manifold fallito: {' '.join(cmd_manifold)}\nstdout: {ris.stdout}\nstderr: {ris.stderr}")

    cmd_simplify = [simplify_bin, "-i", percorso_hull_remeshato, "-o", percorso_output_obj,
                     "-f", str(num_facce_target)]
    ris = subprocess.run(cmd_simplify, capture_output=True, text=True)
    if ris.returncode != 0 or not os.path.isfile(percorso_output_obj):
        raise RuntimeError(f"Simplify fallito: {' '.join(cmd_simplify)}\nstdout: {ris.stdout}\nstderr: {ris.stderr}")

    for tmp in (percorso_hull_grezzo, percorso_hull_remeshato):
        if os.path.isfile(tmp):
            os.remove(tmp)

    print(f"  Mesh iniziale (convex hull remeshato+semplificato): {percorso_output_obj}")
    return percorso_output_obj


# ---------------------------------------------------------------------------
# RICOSTRUZIONE POINT2MESH
# ---------------------------------------------------------------------------

def ricostruisci_point2mesh(pcd: o3d.geometry.PointCloud, nome_run: str, iterations: int = 400,
                             num_facce_mesh_iniziale: int = 2000, manifold_res: int = 20000,
                             export_interval: int = 100):
    """Lancia l'ottimizzazione Point2Mesh su una point cloud gia' pulita e con
    normali stimate, come sottoprocesso nell'ambiente conda 'point2mesh'.

    Ritorna (mesh_o3d, tempo_calcolo_secondi, loss_per_iterazione).
    """
    python_point2mesh = trova_python_ambiente_conda(POINT2MESH_CONDA_ENV)

    cartella_lavoro = os.path.abspath(os.path.join(WORK_DIR, nome_run))
    os.makedirs(cartella_lavoro, exist_ok=True)

    percorso_pc = os.path.join(cartella_lavoro, f"{nome_run}.ply")
    percorso_mesh_iniziale = os.path.join(cartella_lavoro, f"{nome_run}_initmesh.obj")
    cartella_checkpoint = os.path.join(cartella_lavoro, "checkpoints")

    esporta_point_cloud_point2mesh(pcd, percorso_pc)
    genera_mesh_iniziale_convex_hull(pcd, percorso_mesh_iniziale,
                                      num_facce_target=num_facce_mesh_iniziale,
                                      manifold_res=manifold_res)

    cmd = [
        python_point2mesh, "main.py",
        "--input-pc", percorso_pc,
        "--initial-mesh", percorso_mesh_iniziale,
        "--save-path", cartella_checkpoint,
        "--iterations", str(iterations),
        "--export-interval", str(export_interval),
    ]
    print(f"\n[Point2Mesh] Lancio sottoprocesso nell'ambiente '{POINT2MESH_CONDA_ENV}':")
    print(f"  {' '.join(cmd)}")

    pattern_loss = re.compile(r"iter:\s*(\d+)\s*out of:\s*\d+;\s*loss:\s*(-?\d+\.\d+)")
    loss_per_iterazione = []

    inizio = time.time()
    processo = subprocess.Popen(
        cmd, cwd=POINT2MESH_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for riga in processo.stdout:
        print(f"  [point2mesh] {riga}", end="")
        m = pattern_loss.search(riga)
        if m:
            loss_per_iterazione.append((int(m.group(1)), float(m.group(2))))
    processo.wait()
    tempo_calcolo = time.time() - inizio

    if processo.returncode != 0:
        raise RuntimeError(f"Point2Mesh (sottoprocesso) terminato con codice {processo.returncode}")

    percorso_mesh_finale = os.path.join(cartella_checkpoint, "last_recon.obj")
    if not os.path.isfile(percorso_mesh_finale):
        raise FileNotFoundError(f"Point2Mesh non ha prodotto {percorso_mesh_finale}")

    mesh = o3d.io.read_triangle_mesh(percorso_mesh_finale)
    mesh.compute_vertex_normals()
    print(f"[Point2Mesh] Mesh ricostruita: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli "
          f"in {tempo_calcolo:.1f}s")

    return mesh, tempo_calcolo, loss_per_iterazione


def stampa_curva_loss(loss_per_iterazione: list, n_campioni: int = 10):
    """Stampa un riassunto leggibile della curva di loss (non tutte le righe)."""
    if not loss_per_iterazione:
        print("  (nessuna loss catturata dall'output di Point2Mesh)")
        return
    print(f"\n{'=' * 50}")
    print("Curva della loss (Point2Mesh)")
    print(f"{'=' * 50}")
    passo = max(1, len(loss_per_iterazione) // n_campioni)
    for i in range(0, len(loss_per_iterazione), passo):
        it, loss = loss_per_iterazione[i]
        print(f"  iter {it:5d}: loss = {loss:.4f}")
    ultima_it, ultima_loss = loss_per_iterazione[-1]
    print(f"  iter {ultima_it:5d}: loss = {ultima_loss:.4f}  (ultima)")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    pcd = carica_point_cloud(INPUT_CSV, classi_da_escludere=ESCLUDI_CLASSI,
                              altitude_min=ALTITUDE_MIN, altitude_max=ALTITUDE_MAX,
                              escludi_zona_x=ESCLUDI_ZONA_X, escludi_zona_z=ESCLUDI_ZONA_Z)
    pcd = pulisci_point_cloud(pcd, voxel_size=VOXEL_SIZE, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    pcd = subsample_aggressivo(pcd, MAX_PUNTI_POINT2MESH)
    pcd = stima_normali(pcd)

    nome_run = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    mesh, tempo_calcolo, loss_per_iterazione = ricostruisci_point2mesh(
        pcd, nome_run, iterations=ITERATIONS,
        num_facce_mesh_iniziale=INITIAL_MESH_FACES, manifold_res=MANIFOLD_RES,
        export_interval=EXPORT_INTERVAL,
    )
    stampa_curva_loss(loss_per_iterazione)

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)
    print(f"\nMesh salvata in: {OUTPUT_MESH}")

    metriche = valuta_mesh(mesh, pcd, tempo_calcolo=tempo_calcolo, nome_metodo="point2mesh",
                            max_punti_fedelta=MAX_PUNTI_FEDELTA,
                            calcola_autointersezioni=CALCOLA_AUTOINTERSEZIONI)
    stampa_metriche(metriche)
    salva_metriche_csv(metriche, percorso_csv=METRICHE_CSV,
                        input_csv=INPUT_CSV, voxel_size=VOXEL_SIZE, escludi_classi=ESCLUDI_CLASSI)


if __name__ == "__main__":
    main()
