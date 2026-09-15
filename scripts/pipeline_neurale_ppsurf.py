"""
Pipeline neurale di ricostruzione 3D da point cloud (sonar) con PPSurf
============================================================================

PPSurf (https://github.com/cg-tuwien/ppsurf) stima una signed distance
function da una point cloud tramite una rete patch-based (combina un prior
globale a convoluzioni di punti con un prior locale su patch), poi estrae
la mesh con Marching Cubes. A differenza di Point2Mesh (deformazione di un
singolo convex hull, adatto solo a forme compatte genus-0), PPSurf lavora
localmente patch per patch -- strutturalmente adatto a superfici APERTE e
allungate come uno scan sonar multibeam "a nastro".

Nota architetturale (stessa di pipeline_neurale_point2mesh.py): questo
script gira nell'ambiente conda "ricostruzione3d" (Open3D, per riusare
mesh_builder.py), mentre PPSurf richiede l'ambiente separato "pps"
(Python 3.11, PyTorch 2.4 + CUDA 12.4, PyTorch Lightning, PyTorch Geometric).
Il cuore della ricostruzione (repo_esterni/ppsurf/pps.py) viene quindi
lanciato come sottoprocesso con l'interprete dell'ambiente "pps".

Parametri scoperti a mano durante lo smoke test (passo 4), necessari per
girare sulla GPU da 4GB VRAM senza andare in CUDA OutOfMemoryError -- la
fase di marching cubes valuta la rete su una griglia 3D densa ed e' quella
che consuma piu' memoria, indipendentemente dal numero di punti in input:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (riduce la
    frammentazione dell'allocatore CUDA di PyTorch)
  - --model.init_args.gen_resolution_global 129 (risoluzione della griglia
    di marching cubes; il default e' 257 e va in OOM anche da solo)
  - --model.init_args.rec_batch_size 10000 (quanti punti della griglia
    vengono valutati per volta dalla rete; il default e' 50000 e va in OOM
    anche con la risoluzione ridotta a 129)

Dipendenze:
    ambiente "ricostruzione3d": open3d, numpy, pandas (vedi mesh_builder.py)
    ambiente "pps": vedi repo_esterni/ppsurf/pps.yml
    modello pre-addestrato ppsurf_50nn (scaricato automaticamente da pps.py
        rec al primo utilizzo, in repo_esterni/ppsurf/models/)

Uso:
    conda activate ricostruzione3d
    cd scripts
    python pipeline_neurale_ppsurf.py
"""

import glob
import json
import os
import subprocess
import time

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
OUTPUT_MESH = "../output/mesh/mesh_ppsurf.ply"      # percorso file mesh di output
METRICHE_CSV = "../output/metriche_ricostruzione.csv"  # file dove accumulare le metriche di ogni run
CALCOLA_AUTOINTERSEZIONI = False            # True = controllo completo (lento su mesh grandi)
MAX_PUNTI_FEDELTA = 50_000                  # numero massimo di punti usati per il calcolo di fedeltà (raycasting)

# --- Parametri specifici di PPSurf ---
# Il README di PPSurf consiglia di sub-campionare point cloud molto grandi a
# ~250k punti; il nostro dataset pulito (~197k punti) rientra gia' sotto
# questa soglia, quindi di default NON viene applicato alcun subsample
# aggressivo (a differenza di Point2Mesh, dove era indispensabile) -- questo
# e' uno dei vantaggi strutturali di un metodo patch-based. MAX_PUNTI_PPSURF
# resta comunque come rete di sicurezza per dataset futuri piu' grandi.
MAX_PUNTI_PPSURF = 250_000
GEN_RESOLUTION_GLOBAL = 129                 # risoluzione griglia marching cubes (default 257 va in OOM su 4GB)
REC_BATCH_SIZE = 10000                      # chunk size valutazione griglia (default 50000 va in OOM su 4GB)

REPO_ESTERNI = "../repo_esterni"
PPSURF_DIR = os.path.join(REPO_ESTERNI, "ppsurf")
PPSURF_CONDA_ENV = "pps"                    # nome dell'ambiente conda dedicato a PPSurf
WORK_DIR = "../output/ppsurf_tmp"           # point cloud/risultati temporanei scambiati col sottoprocesso


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
        f"conda env create -f {PPSURF_DIR}/pps.yml"
    )


def sottocampiona_se_necessario(pcd: o3d.geometry.PointCloud, max_punti: int) -> o3d.geometry.PointCloud:
    """Rete di sicurezza per dataset molto grandi (soglia suggerita dal README di PPSurf).
    Con il nostro dataset attuale (~197k punti dopo pulisci_point_cloud) non scatta."""
    n = len(pcd.points)
    if n <= max_punti:
        return pcd
    frazione = max_punti / n
    pcd_sub = pcd.random_down_sample(frazione)
    print(f"  Subsample di sicurezza per PPSurf: {n} -> {len(pcd_sub.points)} punti "
          f"(oltre la soglia di {max_punti} consigliata dal README)")
    return pcd_sub


def esporta_point_cloud_ppsurf(pcd: o3d.geometry.PointCloud, percorso_output_ply: str):
    """PPSurf legge i .ply tramite trimesh e usa solo mesh.vertices (vedi
    occupancy_data_module.py: 'pts = np.array(mesh.vertices)') -- le normali
    vengono SEMPRE ignorate, indipendentemente dal formato di input (anche per
    .xyz la documentazione dice esplicitamente che le colonne oltre le prime 3
    vengono scartate). stima_normali() viene comunque chiamata a monte nella
    pipeline per restare coerenti con le altre pipeline di confronto (stesso
    preprocessing), ma il suo output non viene usato da PPSurf."""
    os.makedirs(os.path.dirname(percorso_output_ply), exist_ok=True)
    pcd_solo_punti = o3d.geometry.PointCloud()
    pcd_solo_punti.points = pcd.points
    o3d.io.write_point_cloud(percorso_output_ply, pcd_solo_punti)
    print(f"  Point cloud esportata per PPSurf: {percorso_output_ply} ({len(pcd.points)} punti)")


# ---------------------------------------------------------------------------
# RICOSTRUZIONE PPSURF
# ---------------------------------------------------------------------------

def ricostruisci_ppsurf(pcd: o3d.geometry.PointCloud, nome_run: str,
                         gen_resolution_global: int = 129, rec_batch_size: int = 10000):
    """Lancia la ricostruzione PPSurf (comando 'rec', con il modello pre-addestrato
    50NN) come sottoprocesso nell'ambiente conda 'pps'.

    Ritorna (mesh_o3d, tempo_calcolo_secondi).
    """
    python_ppsurf = trova_python_ambiente_conda(PPSURF_CONDA_ENV)

    cartella_lavoro = os.path.abspath(os.path.join(WORK_DIR, nome_run))
    os.makedirs(cartella_lavoro, exist_ok=True)

    percorso_pc = os.path.join(cartella_lavoro, f"{nome_run}.ply")
    cartella_output = os.path.join(cartella_lavoro, "results")

    esporta_point_cloud_ppsurf(pcd, percorso_pc)

    cmd = [
        python_ppsurf, "pps.py", "rec", percorso_pc, cartella_output,
        "--model.init_args.gen_resolution_global", str(gen_resolution_global),
        "--model.init_args.rec_batch_size", str(rec_batch_size),
    ]

    # IMPORTANTE: questa variabile va passata esplicitamente nell'env del
    # sottoprocesso (non basta averla esportata nel terminale interattivo,
    # altrimenti sparisce quando lo script gira in automatico/in background) --
    # senza, la fase di marching cubes va in CUDA OutOfMemoryError sulla GPU
    # da 4GB anche con rec_batch_size gia' ridotto (verificato nello smoke test).
    env_sottoprocesso = os.environ.copy()
    env_sottoprocesso["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"\n[PPSurf] Lancio sottoprocesso nell'ambiente '{PPSURF_CONDA_ENV}' "
          f"(puo' richiedere diversi minuti)...")
    print(f"  {' '.join(cmd)}")

    inizio = time.time()
    risultato = subprocess.run(cmd, cwd=PPSURF_DIR, env=env_sottoprocesso,
                                capture_output=True, text=True)
    tempo_calcolo = time.time() - inizio

    if risultato.returncode != 0:
        print("--- ultime righe di stdout ---")
        print(risultato.stdout[-3000:])
        print("--- ultime righe di stderr ---")
        print(risultato.stderr[-3000:])
        raise RuntimeError(f"PPSurf (sottoprocesso) terminato con codice {risultato.returncode}")

    print(f"[PPSurf] Sottoprocesso completato in {tempo_calcolo:.1f}s")

    # PPSurf salva l'output con un path del tipo:
    #   {out_dir}/{basename(in_file)}/{basename(in_file)}.ply
    # (nome del file di input ripetuto come sotto-cartella E con estensione
    # raddoppiata -- comportamento del codice sorgente, vedi source/poco_model.py
    # predict_step(), non documentato nel README). Uso glob invece di ricostruire
    # il path a mano per restare robusti a eventuali differenze di dettaglio.
    pattern_output = os.path.join(cartella_output, os.path.basename(percorso_pc), "*.ply")
    candidati = glob.glob(pattern_output)
    if not candidati:
        raise FileNotFoundError(f"Nessun file .ply trovato con il pattern: {pattern_output}")
    percorso_mesh_finale = candidati[0]

    mesh = o3d.io.read_triangle_mesh(percorso_mesh_finale)
    mesh.compute_vertex_normals()
    print(f"[PPSurf] Mesh ricostruita: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli")

    return mesh, tempo_calcolo


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    pcd = carica_point_cloud(INPUT_CSV, classi_da_escludere=ESCLUDI_CLASSI,
                              altitude_min=ALTITUDE_MIN, altitude_max=ALTITUDE_MAX,
                              escludi_zona_x=ESCLUDI_ZONA_X, escludi_zona_z=ESCLUDI_ZONA_Z)
    pcd = pulisci_point_cloud(pcd, voxel_size=VOXEL_SIZE, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    pcd = sottocampiona_se_necessario(pcd, MAX_PUNTI_PPSURF)
    pcd = stima_normali(pcd)  # non usato da PPSurf (vedi esporta_point_cloud_ppsurf), mantenuto per coerenza

    nome_run = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    mesh, tempo_calcolo = ricostruisci_ppsurf(
        pcd, nome_run, gen_resolution_global=GEN_RESOLUTION_GLOBAL, rec_batch_size=REC_BATCH_SIZE,
    )

    os.makedirs(os.path.dirname(OUTPUT_MESH), exist_ok=True)
    o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)
    print(f"\nMesh salvata in: {OUTPUT_MESH}")

    metriche = valuta_mesh(mesh, pcd, tempo_calcolo=tempo_calcolo, nome_metodo="ppsurf",
                            max_punti_fedelta=MAX_PUNTI_FEDELTA,
                            calcola_autointersezioni=CALCOLA_AUTOINTERSEZIONI)
    stampa_metriche(metriche)
    salva_metriche_csv(metriche, percorso_csv=METRICHE_CSV,
                        input_csv=INPUT_CSV, voxel_size=VOXEL_SIZE, escludi_classi=ESCLUDI_CLASSI)


if __name__ == "__main__":
    main()
