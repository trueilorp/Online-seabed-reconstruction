"""
Pipeline neurale di ricostruzione 3D da point cloud (sonar) con SA-ConvONet
============================================================================

SA-ConvONet (https://github.com/tangjiapeng/SA-ConvONet) parte da un modello
Convolutional Occupancy Network pre-addestrato (ConvONet) e ne ottimizza i
pesi in modo "sign-agnostic" per-scena a test-time. Qui l'ottimizzazione e'
opzionale (sa_convonet.ottimizzazione_online in config): se disattivata si usa
generate.py, cioe' ConvONet pre-addestrato senza adattamento; se attivata
generate_optim_largescene.py, che adatta la rete alla scena prima di
estrarre la mesh (molto piu' lento). Usiamo la variante
`pointcloud_crop` (encoder a griglia 3D con finestra scorrevole, modello
`room_grid64`): a differenza dei modelli ConvONet "stanza" (bounding box
fissa, coordinate normalizzate), questa variante e' pensata per scene reali
di grandi dimensioni con unita' metriche vere -- lo stesso identikit della
nostra scansione sonar (superficie aperta, transetto su larga scala).

Nota architetturale (stessa di scripts/point2mesh.py e scripts/ppsurf.py):
questo script gira nell'ambiente conda "ricostruzione3d" (Open3D, per
riusare utils/mesh.py), mentre SA-ConvONet richiede l'ambiente separato
"sa_conet" (Python 3.6, PyTorch 1.4.0 + CUDA 10.1, torch_scatter). Il
preprocessing dataset (helpers.../make_cropscene_dataset2.py) e la
ricostruzione vera e propria (generate.py o generate_optim_largescene.py) vengono quindi
lanciati come sottoprocessi nell'ambiente "sa_conet".

Nota IMPORTANTE sulla PyTorch dell'ambiente "sa_conet": va installata dalla
wheel ufficiale pip (non dal canale conda "pytorch"), altrimenti la ABI
C++ non combacia con la wheel precompilata di torch_scatter (simboli
`std::string` vecchia ABI vs nuova ABI di libstdc++) e qualsiasi script
che la importa va in crash con un "undefined symbol" all'avvio. Fix
verificato:
    pip install --force-reinstall --no-deps \\
        https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl
    pip install --force-reinstall --no-deps \\
        https://data.pyg.org/whl/torch-1.4.0+cu101/torch_scatter-2.0.4-cp36-cp36m-linux_x86_64.whl

I parametri di questa pipeline sono in config/config.yaml (sezione
sa_convonet); l'esecuzione avviene tramite reconstruct_single_dataset.py / evaluate.py
(via utils/pipeline.py) nella root del progetto.
"""

import glob
import json
import os
import subprocess
import time

import numpy as np
import open3d as o3d
import yaml


# ---------------------------------------------------------------------------
# UTILITA'
# ---------------------------------------------------------------------------

def trova_python_ambiente_conda(nome_ambiente: str) -> str:
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
        f"conda env create -f environment.yaml (dentro la cartella sa_convonet_dir)"
    )


def esporta_point_cloud_sa_convonet(pcd: o3d.geometry.PointCloud, percorso_output_ply: str):
    """make_cropscene_dataset2.py (script di SA-ConvONet) legge il .ply con
    plyfile cercando le proprieta' x,y,z,nx,ny,nz -- richiede quindi le
    normali gia' stimate, senza bisogno di colori."""
    if not pcd.has_normals():
        raise ValueError("La point cloud non ha normali stimate: chiama stima_normali() prima")
    os.makedirs(os.path.dirname(percorso_output_ply), exist_ok=True)
    pcd_senza_colori = o3d.geometry.PointCloud()
    pcd_senza_colori.points = pcd.points
    pcd_senza_colori.normals = pcd.normals
    o3d.io.write_point_cloud(percorso_output_ply, pcd_senza_colori)
    print(f"  Point cloud esportata per SA-ConvONet: {percorso_output_ply} ({len(pcd.points)} punti)")


# ---------------------------------------------------------------------------
# RICOSTRUZIONE SA-CONVONET
# ---------------------------------------------------------------------------

SOGLIA_ONLINE = 0.2  # soglia di estrazione con ottimizzazione online (= test.threshold del modello)


def _checkpoint_locale(checkpoint_url: str) -> str:
    """Percorso locale dei pesi pre-addestrati nella cache di torch.

    generate_optim_largescene.py ricarica i pesi con torch.load(os.path.join(out_dir,
    model_file)), che con un URL fallisce: serve il file gia' scaricato. La cache
    viene riempita dalla prima esecuzione di generate.py (ottimizzazione_online: false).
    """
    if not checkpoint_url:
        raise ValueError("Ottimizzazione online: test.model_file non trovato nel config SA-ConvONet")
    if os.path.isfile(checkpoint_url):
        return os.path.abspath(checkpoint_url)
    torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
    nome = os.path.basename(checkpoint_url)
    for sottocartella in ("checkpoints", os.path.join("hub", "checkpoints")):
        percorso = os.path.join(torch_home, sottocartella, nome)
        if os.path.isfile(percorso):
            return percorso
    raise FileNotFoundError(
        f"Pesi {nome} non trovati nella cache di torch ({torch_home}): lancia una volta "
        f"SA-ConvONet con ottimizzazione_online: false per scaricarli")

def ricostruisci_sa_convonet(pcd: o3d.geometry.PointCloud, nome_run: str,
                              sa_convonet_dir: str, conda_env: str, work_dir: str,
                              config_file: str, unit_size: float = 0.002,
                              query_vol_size: int = 25, ottimizzazione_online: bool = False,
                              online_iterazioni: int = 2000, online_passo_lr: int = 800,
                              online_batch_size: int = 3, checkpoint_url: str = None):
    """Prepara il dataset nel formato atteso da SA-ConvONet (sottoprocesso su
    make_cropscene_dataset2.py, che centra/scala la point cloud in unita'
    coerenti col modello pre-addestrato) e poi lancia generate.py sulla
    config indicata (sottoprocesso nell'ambiente conda 'sa_conet').

    unit_size / query_vol_size: dimensione di ogni crop nella generazione a
        finestre scorrevoli (crop = unit_size * query_vol_size, in unita'
        normalizzate). Il numero totale di crop scala con l'estensione reale
        della scena divisa per questo valore -- per un transetto sonar su
        larga scala (centinaia di metri) i default del repo (tarati su
        stanze Matterport, 10-20m) generano migliaia di crop e run da ore;
        valori piu' alti riducono i crop (piu' veloce) a scapito della
        risoluzione della mesh.

    ottimizzazione_online: se True usa generate_optim_largescene.py
        (adattamento sign-agnostic per-scena, "SA") invece di generate.py.
        online_iterazioni / online_passo_lr / online_batch_size sovrascrivono
        test_optim.n_iter / n_step / batch_size. Ogni online_passo_lr iterazioni
        il learning rate decade E viene generata una mesh intermedia completa
        (costo di un intero generate.py): conviene online_passo_lr non troppo piccolo.
    checkpoint_url: pesi pre-addestrati (URL o file); se None si legge
        test.model_file da config_file. Con l'ottimizzazione online serve il
        file locale nella cache di torch (vedi _checkpoint_locale).

    Ritorna (mesh_o3d, tempo_calcolo_secondi).
    """
    python_sa_convonet = trova_python_ambiente_conda(conda_env)

    cartella_lavoro = os.path.abspath(os.path.join(work_dir, nome_run))
    cartella_grezza = os.path.join(cartella_lavoro, "raw", nome_run)
    cartella_normalizzata = os.path.join(cartella_lavoro, "normalize")

    percorso_ply = os.path.join(cartella_grezza, "pointcloud.ply")
    esporta_point_cloud_sa_convonet(pcd, percorso_ply)

    cmd_prepara = [
        python_sa_convonet, "scripts/dataset_matterport/make_cropscene_dataset2.py",
        "--in_folder", os.path.join(cartella_lavoro, "raw"),
        "--out_folder", cartella_normalizzata,
        "--do_norm",
    ]
    print(f"\n[SA-ConvONet] Preparazione dataset nell'ambiente '{conda_env}':")
    print(f"  {' '.join(cmd_prepara)}")
    risultato = subprocess.run(cmd_prepara, cwd=sa_convonet_dir, capture_output=True, text=True)
    if risultato.returncode != 0:
        print("--- ultime righe di stdout ---")
        print(risultato.stdout[-3000:])
        print("--- ultime righe di stderr ---")
        print(risultato.stderr[-3000:])
        raise RuntimeError(f"Preparazione dataset SA-ConvONet fallita (codice {risultato.returncode})")

    # config_file (es. configs/pointcloud_crop/sonar.yaml) ha un data.path
    # statico che non conosce la cartella di lavoro temporanea di questo run
    # (fuori dal repo, in output/sa_convonet_tmp/<nome_run>/, come per
    # point2mesh/ppsurf) -- genero quindi un config che eredita da
    # config_file ma sovrascrive path dinamici con quelli reali di questo run.
    config_run = os.path.join(cartella_lavoro, "run_config.yaml")
    contenuto_config = {
        "inherit_from": config_file,
        "data": {
            "path": cartella_normalizzata,
            "unit_size": unit_size,
            "query_vol_size": query_vol_size,
        },
        "training": {"out_dir": os.path.join(cartella_lavoro, "out")},
    }
    if ottimizzazione_online:
        if checkpoint_url is None:
            with open(os.path.join(sa_convonet_dir, config_file)) as fc:
                checkpoint_url = (yaml.safe_load(fc).get("test") or {}).get("model_file")
        contenuto_config["test"] = {"model_file": _checkpoint_locale(checkpoint_url)}
        contenuto_config["test_optim"] = {
            "n_iter": online_iterazioni,
            "n_step": online_passo_lr,
            "batch_size": online_batch_size,
            # una sola soglia (= test.threshold di generate.py): lo script genera una mesh
            # completa per ogni soglia a ogni passo, con le 3 di default il costo triplica
            "threshold": [SOGLIA_ONLINE],
        }
    with open(config_run, "w") as f:
        yaml.safe_dump(contenuto_config, f)

    script_generazione = "generate_optim_largescene.py" if ottimizzazione_online else "generate.py"
    cmd_generate = [python_sa_convonet, "-u", script_generazione, config_run]
    print(f"\n[SA-ConvONet] Lancio ricostruzione nell'ambiente '{conda_env}' "
          f"(puo' richiedere diversi minuti)...")
    print(f"  {' '.join(cmd_generate)}")

    # Streaming dell'output riga per riga invece di subprocess.run(capture_output=True):
    # la generazione a finestre scorrevoli puo' richiedere a lungo (migliaia di crop
    # su scene molto estese), e senza streaming non si vede alcun progresso (tqdm
    # incluso) finche' il processo non termina -- "-u" disabilita il buffering di
    # Python nel sottoprocesso cosi' i print arrivano subito, non solo a fine run.
    inizio = time.time()
    righe_output = []
    processo = subprocess.Popen(
        cmd_generate, cwd=sa_convonet_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for riga in processo.stdout:
        print(f"  [sa_convonet] {riga}", end="")
        righe_output.append(riga)
    processo.wait()
    tempo_calcolo = time.time() - inizio

    if processo.returncode != 0:
        raise RuntimeError(f"SA-ConvONet (sottoprocesso) terminato con codice {processo.returncode}")

    print(f"[SA-ConvONet] Sottoprocesso completato in {tempo_calcolo:.1f}s")

    # generate.py salva la mesh in:
    #   {out_dir}/{generation_dir}/meshes/{categoria}/{modelname}.off
    # con out_dir = cartella_lavoro/out (impostato sopra in config_run),
    # categoria = 'large_scenes' (nome cartella creato da
    # make_cropscene_dataset2.py) e modelname = nome_run. Uso glob per
    # restare robusti a eventuali differenze di dettaglio nel config.
    # con l'ottimizzazione online la mesh adattata e' <nome_run>_final_th<soglia>.off
    # (<nome_run>.off e' quella prima dell'adattamento, generata comunque dallo script)
    nome_mesh = f"{nome_run}_final_th{SOGLIA_ONLINE}.off" if ottimizzazione_online else f"{nome_run}.off"
    pattern_output = os.path.join(cartella_lavoro, "out", "**", nome_mesh)
    candidati = glob.glob(pattern_output, recursive=True)
    if not candidati:
        raise FileNotFoundError(f"Nessun file .off trovato con il pattern: {pattern_output}")
    percorso_mesh_finale = max(candidati, key=os.path.getmtime)

    mesh = o3d.io.read_triangle_mesh(percorso_mesh_finale)

    # make_cropscene_dataset2.py normalizza la point cloud in input (centra su
    # 'loc' e scala per 'scale', salvati dentro pointcloud.npz) -- generate.py
    # produce quindi la mesh nello stesso spazio normalizzato, non in metri
    # reali. Va invertita la stessa trasformazione, altrimenti la mesh resta
    # in un cubo unitario mentre la point cloud originale e' a coordinate
    # reali (differenza di migliaia di metri nelle metriche di fedelta').
    percorso_npz = os.path.join(cartella_normalizzata, "large_scenes", nome_run, "pointcloud.npz")
    normalizzazione = np.load(percorso_npz)
    loc, scale = normalizzazione["loc"], normalizzazione["scale"]
    vertici_reali = np.asarray(mesh.vertices) * scale + loc
    mesh.vertices = o3d.utility.Vector3dVector(vertici_reali)

    mesh.compute_vertex_normals()
    print(f"[SA-ConvONet] Mesh ricostruita: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli "
          f"(riportata in coordinate reali: loc={loc}, scale={scale:.3f})")

    return mesh, tempo_calcolo
