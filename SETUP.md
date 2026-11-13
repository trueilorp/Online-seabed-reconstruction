# Setup on a new machine

How to get the project running on a freshly cloned Linux machine. The git
repository contains only the code: the neural repositories
(`neural_methods/`), the data (`data/synthetic/`) and the results (`output/`)
are excluded and must be recreated.

Run every command from the repository root unless stated otherwise.

## 1. Prerequisites

| Requirement | Check |
|---|---|
| Linux, or Windows with WSL2 | `uname -a` |
| Miniconda, with `conda` in `PATH` | `conda --version` |
| NVIDIA GPU with a recent driver (PPSurf runs on CUDA) | `nvidia-smi` lists the GPU and a CUDA version ≥ 12.4 |
| Disk space: ~20 GB | see below |

Approximate disk usage on the reference machine:

| Item | Size |
|---|---|
| conda environment `pps` | 7.3 GB |
| conda environment `ricostruzione3d` | 1.6 GB |
| `data/synthetic/` (all splits, 102 terrains) | 7.0 GB |
| `neural_methods/ppsurf/` (with pretrained weights) | 180 MB |
| one full evaluation run on `test+val` | 117 MB – 1.6 GB |

The GPU memory needed depends on the PPSurf settings; see
[section 6](#6-configuration-for-low-vram).

## 2. Neural repository and trimesh patch

```bash
mkdir -p neural_methods
git clone https://github.com/cg-tuwien/ppsurf.git neural_methods/ppsurf

# PPSurf fix for trimesh >= 5 (remove_degenerate_faces / remove_duplicate_faces were removed)
sed -i 's/mesh.remove_degenerate_faces()/mesh.update_faces(mesh.nondegenerate_faces())/; s/mesh.remove_duplicate_faces()/mesh.update_faces(mesh.unique_faces())/' neural_methods/ppsurf/source/base/mesh.py
```

Check that the patch was applied:

```bash
grep -n "nondegenerate_faces\|unique_faces" neural_methods/ppsurf/source/base/mesh.py
# expected: two lines with mesh.update_faces(...)
```

The pretrained PPSurf weights (`ppsurf_50nn`) are downloaded automatically by
`pps.py` on the first reconstruction, into `neural_methods/ppsurf/models/`.

## 3. Conda environments

Two environments are used:

| Name | Purpose | Referenced by |
|---|---|---|
| `ricostruzione3d` | main pipeline (Open3D, pandas, matplotlib) | `env_setup.sh` |
| `pps` | PPSurf (PyTorch, CUDA) | `ppsurf.conda_env` in `config/config.yaml` |

```bash
# main pipeline
conda create -n ricostruzione3d python=3.10 -y
conda activate ricostruzione3d
pip install -r requirements.txt
pip install psutil          # optional: without it, the RAM peak (ram_picco_mb) is not recorded

# PPSurf
conda env create -f neural_methods/ppsurf/pps.yml
conda install -n pps "pytorch::pytorch=2.4.1" "pytorch::pytorch-cuda=12.4" -c pytorch -c nvidia
```

The second command for `pps` is needed because `pps.yml` does not pin the
build, and conda may install a **CPU-only** PyTorch from conda-forge (PPSurf
then fails with `No supported gpu backend found!`).

Check both environments:

```bash
conda run -n ricostruzione3d python -c "import open3d, pandas, yaml; print(open3d.__version__)"
# expected: 0.19.0 or later

conda run -n pps python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
# expected: 12.4 True
```

## 4. Synthetic data

In the `ricostruzione3d` environment:

```bash
python utils/generate_seabed_heightmaps.py --with-structures --n-per-family 17
python utils/simulate_multibeam_scans.py
```

`--n-per-family 17` reproduces the current dataset (72 / 18 / 12 terrains in
train / val / test). With the default seed (`--seed 0`) the terrains are
identical. The first command ends with:

```
Generati 102 terreni -> data/synthetic  (train 72, val 18, test 12)
```

The second command reads `data/synthetic/manifest.csv` and writes, in each
`data/synthetic/<split>/<id>/` folder, `<id>_scan.csv`,
`<id>_ground_truth.obj` and `<id>_scan.json`. Its parameters (sensor, survey
path, which terrains) are constants at the top of the script.

All results in the README use the `test+val` split, the union of `test` and
`val`. Create it with:

```bash
mkdir -p data/synthetic/test+val
cp -r data/synthetic/test/* data/synthetic/val/* data/synthetic/test+val/
```

Check the folder sizes:

```bash
for s in train val test test+val; do echo "$s: $(ls data/synthetic/$s | wc -l)"; done
# expected: train: 72, val: 18, test: 12, test+val: 30
ls data/synthetic/test+val/flat_sand_015
# expected among the files: flat_sand_015_scan.csv  flat_sand_015_ground_truth.obj
```

## 5. Verify the installation

```bash
source env_setup.sh            # activates ricostruzione3d (the WAYLAND fix only matters on WSLg)

# one terrain per family, one method at a time: a failure points to the environment at fault
python evaluate.py --split val --limit 1 --methods poisson
python evaluate.py --split val --limit 1 --methods ppsurf
```

Each command creates `output/evaluation/<date_time>_val/`. Check that:

- `metriche_per_terreno.csv` has 6 rows, with an empty `errore` (error)
  column;
- `meshes/` contains 6 `.ply` files.

Poisson takes a few seconds per terrain. PPSurf takes 38–43 s per terrain on a
GTX TITAN X with `rec_batch_size=50000`, and longer with the default 10000; the
first run also downloads the weights.

## 6. Configuration for low VRAM

The defaults in `config/config.yaml` are sized for a 4 GB GPU (GTX 1650):

| Setting | Value | PPSurf default | Effect |
|---|---|---|---|
| `ppsurf.gen_resolution_global` | 129 | 257 | marching-cubes grid; 257 runs out of memory on 4 GB |
| `ppsurf.rec_batch_size` | 10000 | 50000 | grid points evaluated per batch; affects speed and memory, not the result |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | set by `scripts/ppsurf.py` | – | reduces CUDA allocator fragmentation |

Peak VRAM measured with these settings: ~1.9 GB at batch 10000, ~8.6–8.7 GB at
batch 50000 (grid 129 in both cases).

With more memory, raise the batch for one run without editing the config:

```bash
python evaluate.py --methods ppsurf --set ppsurf.rec_batch_size=50000
```

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'Trimesh' object has no attribute 'remove_degenerate_faces'` at the end of a PPSurf prediction | trimesh ≥ 5 removed these methods; PPSurf still calls them | apply the patch in [section 2](#2-neural-repository-and-trimesh-patch) |
| PPSurf fails with `No supported gpu backend found!` | `pps` has a CPU-only PyTorch build from conda-forge | reinstall PyTorch with CUDA, as in [section 3](#3-conda-environments), and check `torch.cuda.is_available()` |
| PyTorch errors on a Maxwell GPU (e.g. GTX TITAN X) | PyTorch ≥ 2.8 builds no longer support Maxwell | keep PyTorch 2.4.1 in `pps`; do not upgrade it to ≥ 2.8 |
| `CUDA out of memory` in PPSurf; the terrain is marked as an error | grid or batch too large for the GPU | lower `ppsurf.rec_batch_size` first (it does not change the result), then `ppsurf.gen_resolution_global`; see [section 6](#6-configuration-for-low-vram) |
| The Open3D window does not appear on WSLg, or `screenshot_run.py` reports `Finestra Open3D non creata` ("window not created") | GLFW tries Wayland and the window cannot be created | `visualize_mesh.py` and `screenshot_run.py` force X11 themselves; for other scripts, run `source env_setup.sh`, which empties `WAYLAND_DISPLAY` |
| matplotlib crashes while saving a figure on a machine without a display | an interactive backend is selected | call `matplotlib.use("Agg")` **before** importing `pyplot` (the plotting scripts in `visualization/` already do) |
| A PPSurf terrain fails with `TimeoutError: PPSurf non ha finito entro 1200 s` ("did not finish within") | the subprocess hung (observed once with CPU at 100% and GPU idle) | the subprocess is killed automatically after `ppsurf.timeout_s`; rerun the terrain |
| The WSL virtual disk (`.vhdx`) keeps growing after deleting files | the `.vhdx` grows but does not shrink by itself | from Windows: `wsl --shutdown`, then `Optimize-VHD` (Hyper-V module) or `diskpart` → `compact vdisk` |
