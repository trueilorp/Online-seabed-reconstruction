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
git clone https://github.com/tangjiapeng/SA-ConvONet.git neural_methods/SA-ConvONet

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
python utils/make_saconvonet_dataset.py                       # dataset di fine-tuning SA-ConvONet
```

La scansione Unity `data/sonar/Terrain_unity.csv` (con `Terrain_ground_truth.obj`),
usata come input di default in `config/config.yaml`, non si rigenera: va copiata
dal PC di sviluppo.

## 4. Verifica

```bash
source env_setup.sh            # attiva ricostruzione3d (il fix WAYLAND serve solo su WSLg)

# un terreno per famiglia, un metodo alla volta: isola subito l'ambiente che non va
python evaluate.py --limit 1 --methods poisson
python evaluate.py --limit 1 --methods ppsurf
python evaluate.py --limit 1 --methods sa_convonet
```

## 5. SA-ConvONet (da verificare su questa macchina)

Passi usati sul PC di sviluppo, non ancora provati qui:

```bash
# config del sonar, richiesto da config/config.yaml (non esiste nella repo originale)
cp setup/sa_convonet_sonar.yaml neural_methods/SA-ConvONet/configs/pointcloud_crop/sonar.yaml

# ambiente: togliere pytorch, torchvision e torch_scatter da environment.yaml, poi
conda env create -f neural_methods/SA-ConvONet/environment.yaml   # crea "sa_conet"
conda activate sa_conet
pip install --no-deps https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl
pip install --no-deps https://data.pyg.org/whl/torch-1.4.0+cu101/torch_scatter-2.0.4-cp36-cp36m-linux_x86_64.whl
cd neural_methods/SA-ConvONet && python setup.py build_ext --inplace && cd ../..
```

torch e torch_scatter vanno presi entrambi dalle wheel pip: con il PyTorch del
canale conda, torch_scatter va in crash all'import (`undefined symbol`, ABI C++
diversa; vedi README → Note tecniche).

## Note

- **GPU Maxwell (es. GTX TITAN X)**: non aggiornare PyTorch a ≥ 2.8 in `pps`,
  le build recenti non supportano più queste schede.
- **Parametri tarati per 4 GB di VRAM**: con più memoria si possono alzare
  `ppsurf.gen_resolution_global` (default PPSurf 257) e `ppsurf.rec_batch_size`
  (default PPSurf 50000) in `config/config.yaml`.
- **Via SSH**: lasciare `visualizza: false` e lanciare i run lunghi dentro `tmux`.
