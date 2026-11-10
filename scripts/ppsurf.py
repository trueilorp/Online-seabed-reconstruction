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

Nota architetturale (stessa di scripts/point2mesh.py): questo script gira
nell'ambiente conda "ricostruzione3d" (Open3D, per riusare utils/mesh.py),
mentre PPSurf richiede l'ambiente separato "pps" (Python 3.11, PyTorch 2.4
+ CUDA 12.4, PyTorch Lightning, PyTorch Geometric). Il cuore della
ricostruzione (helpers/ppsurf/pps.py) viene quindi lanciato come
sottoprocesso con l'interprete dell'ambiente "pps".

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
    ambiente "ricostruzione3d": open3d, numpy, pandas (vedi utils/mesh.py)
    ambiente "pps": vedi helpers/ppsurf/pps.yml
    modello pre-addestrato ppsurf_50nn (scaricato automaticamente da pps.py
        rec al primo utilizzo, in helpers/ppsurf/models/)

I parametri di questa pipeline sono in config/config.yaml (sezione
ppsurf); l'esecuzione avviene tramite main.py nella root del progetto.
"""

import glob
import json
import os
import signal
import subprocess
import time

import open3d as o3d


# ---------------------------------------------------------------------------
# UTILITA'
# ---------------------------------------------------------------------------

def trova_python_ambiente_conda(nome_ambiente: str, ppsurf_dir: str) -> str:
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
        f"conda env create -f {ppsurf_dir}/pps.yml"
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
    vengono scartate). Per questo la pipeline non stima le normali per PPSurf."""
    os.makedirs(os.path.dirname(percorso_output_ply), exist_ok=True)
    pcd_solo_punti = o3d.geometry.PointCloud()
    pcd_solo_punti.points = pcd.points
    o3d.io.write_point_cloud(percorso_output_ply, pcd_solo_punti)
    print(f"  Point cloud esportata per PPSurf: {percorso_output_ply} ({len(pcd.points)} punti)")


# ---------------------------------------------------------------------------
# RICOSTRUZIONE PPSURF
# ---------------------------------------------------------------------------

def ricostruisci_ppsurf(pcd: o3d.geometry.PointCloud, nome_run: str,
                         ppsurf_dir: str, conda_env: str, work_dir: str,
                         gen_resolution_global: int = 129, rec_batch_size: int = 10000,
                         timeout_s: float = None, gen_subsample_manifold_iter: int = None,
                         gen_refine_iter: int = None):
    """Lancia la ricostruzione PPSurf (comando 'rec', con il modello pre-addestrato
    50NN) come sottoprocesso nell'ambiente conda 'pps'.

    gen_subsample_manifold_iter: ripetizioni della parte globale della rete su sottoinsiemi dei
        punti (test-time augmentation; default di PPSurf 10). None = default di PPSurf.
    gen_refine_iter: iterazioni di raffinamento delle intersezioni di marching cubes
        (default di PPSurf 10). None = default di PPSurf.
    timeout_s: oltre questo tempo il sottoprocesso (con i suoi worker) viene terminato e si
    solleva un errore, cosi' un PPSurf bloccato non ferma l'intera valutazione. None = nessun limite.

    Ritorna (mesh_o3d, tempo_calcolo_secondi).
    """
    python_ppsurf = trova_python_ambiente_conda(conda_env, ppsurf_dir)

    cartella_lavoro = os.path.abspath(os.path.join(work_dir, nome_run))
    os.makedirs(cartella_lavoro, exist_ok=True)

    percorso_pc = os.path.join(cartella_lavoro, f"{nome_run}.ply")
    cartella_output = os.path.join(cartella_lavoro, "results")

    esporta_point_cloud_ppsurf(pcd, percorso_pc)

    cmd = [
        python_ppsurf, "pps.py", "rec", percorso_pc, cartella_output,
        "--model.init_args.gen_resolution_global", str(gen_resolution_global),
        "--model.init_args.rec_batch_size", str(rec_batch_size),
    ]
    if gen_subsample_manifold_iter is not None:
        cmd += ["--model.init_args.gen_subsample_manifold_iter", str(gen_subsample_manifold_iter)]
    if gen_refine_iter is not None:
        cmd += ["--model.init_args.gen_refine_iter", str(gen_refine_iter)]

    # IMPORTANTE: questa variabile va passata esplicitamente nell'env del
    # sottoprocesso (non basta averla esportata nel terminale interattivo,
    # altrimenti sparisce quando lo script gira in automatico/in background) --
    # senza, la fase di marching cubes va in CUDA OutOfMemoryError sulla GPU
    # da 4GB anche con rec_batch_size gia' ridotto (verificato nello smoke test).
    env_sottoprocesso = os.environ.copy()
    env_sottoprocesso["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"\n[PPSurf] Lancio sottoprocesso nell'ambiente '{conda_env}' "
          f"(puo' richiedere diversi minuti)...")
    print(f"  {' '.join(cmd)}")

    inizio = time.time()
    # sessione propria: in caso di timeout si termina anche i worker del dataloader di PPSurf
    processo = subprocess.Popen(cmd, cwd=ppsurf_dir, env=env_sottoprocesso, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = processo.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(processo.pid, signal.SIGKILL)
        processo.communicate()
        raise TimeoutError(f"PPSurf non ha finito entro {timeout_s:.0f} s: sottoprocesso terminato")
    risultato = subprocess.CompletedProcess(cmd, processo.returncode, stdout, stderr)
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
