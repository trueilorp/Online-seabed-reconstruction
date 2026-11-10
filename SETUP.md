# Setup su una nuova macchina

Procedura per rendere funzionante il progetto su un PC Linux appena clonato.
La repo contiene solo il codice: i repo neurali (`neural_methods/`), i dati
(`data/`) e i risultati (`output/`) sono esclusi da git e vanno ricreati.

Tutti i comandi si lanciano dalla root del progetto. Serve Miniconda con
`conda` nel `PATH`.

## 1. Repo neurali

```bash
mkdir -p neural_methods
git clone https://github.com/cg-tuwien/ppsurf.git neural_methods/ppsurf

# fix PPSurf per trimesh >= 5 (remove_degenerate_faces / remove_duplicate_faces rimossi)
sed -i 's/mesh.remove_degenerate_faces()/mesh.update_faces(mesh.nondegenerate_faces())/; s/mesh.remove_duplicate_faces()/mesh.update_faces(mesh.unique_faces())/' neural_methods/ppsurf/source/base/mesh.py
```

Senza il fix PPSurf fallisce a fine predizione con
`AttributeError: 'Trimesh' object has no attribute 'remove_degenerate_faces'`.

I pesi pre-addestrati di PPSurf (`ppsurf_50nn`) si scaricano da soli al primo utilizzo.

## 2. Ambienti conda

I nomi `ricostruzione3d` e `pps` sono quelli attesi da `config/config.yaml`.

```bash
# pipeline principale (Open3D)
conda create -n ricostruzione3d python=3.10 -y
conda activate ricostruzione3d
pip install -r requirements.txt

# PPSurf
conda env create -f neural_methods/ppsurf/pps.yml
conda install -n pps "pytorch::pytorch=2.4.1" "pytorch::pytorch-cuda=12.4" -c pytorch -c nvidia
```

Il secondo comando per `pps` serve perché `pps.yml` non fissa la build e conda
può prendere PyTorch **solo CPU** da conda-forge (errore di PPSurf:
`No supported gpu backend found!`). Verifica:

```bash
conda run -n pps python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"   # 12.4 True
```

## 3. Dati sintetici

Nell'ambiente `ricostruzione3d` (il seed di default riproduce gli stessi terreni):

```bash
python utils/generate_seabed_heightmaps.py --with-structures   # heightmap + manifest.csv
python utils/simulate_multibeam_scans.py                      # <id>_scan.csv + <id>_ground_truth.obj
```

La cartella `data/synthetic/test+val/` (unione di test e val, usata dai test attuali)
va creata a mano copiando le cartelle dei terreni dei due split.

## 4. Verifica

```bash
source env_setup.sh            # attiva ricostruzione3d (il fix WAYLAND serve solo su WSLg)

# un terreno per famiglia, un metodo alla volta: isola subito l'ambiente che non va
python evaluate.py --split val --limit 1 --methods poisson
python evaluate.py --split val --limit 1 --methods ppsurf
```

## Note

- **GPU Maxwell (es. GTX TITAN X)**: non aggiornare PyTorch a ≥ 2.8 in `pps`,
  le build recenti non supportano più queste schede.
- **Parametri tarati per 4 GB di VRAM**: con più memoria si possono alzare
  `ppsurf.gen_resolution_global` (default PPSurf 257) e `ppsurf.rec_batch_size`
  (default PPSurf 50000) in `config/config.yaml`.
- **Via SSH**: lanciare i run lunghi dentro `tmux`.
