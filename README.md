# Seabed 3D reconstruction from multibeam sonar

This repository reconstructs a 3D mesh of the seabed from multibeam sonar point
clouds and evaluates it against a known ground truth. It compares
**Poisson** surface reconstruction (Open3D) and **PPSurf** (neural,
pretrained), each on the full scan and with a **sliding window** (SW) that
processes the scan in windows of pings, as it would arrive during a survey.

This README covers installation and usage. Method, results and diagnostic
experiments are in the project report, [report.pdf](report.pdf);
the experiment notes are in [ANALYSIS.md](ANALYSIS.md).

## Contents

1. [Installation](#1-installation)
2. [Synthetic data](#2-synthetic-data)
3. [Usage](#3-usage)
4. [Contacts](#contacts)

## 1. Installation

The git repository contains only the code: the neural repository
(`neural_methods/`), the data (`data/synthetic/`) and the results (`output/`)
are excluded and must be recreated. Run every command from the repository
root.

### 1.1 Prerequisites

| Requirement | Check |
|---|---|
| Linux, or Windows with WSL2 | `uname -a` |
| Miniconda, with `conda` in `PATH` | `conda --version` |
| NVIDIA GPU with a recent driver (PPSurf runs on CUDA) | `nvidia-smi` lists the GPU and a CUDA version ≥ 12.4 |
| Disk space: ~20 GB | see below |

| Item | Size on the reference machine |
|---|---|
| conda environment `pps` | 7.3 GB |
| conda environment `ricostruzione3d` | 1.6 GB |
| `data/synthetic/` (all splits, 102 terrains) | 7.0 GB |
| `neural_methods/ppsurf/` (with pretrained weights) | 180 MB |
| one full evaluation run on `test+val` | 117 MB – 1.6 GB |

### 1.2 PPSurf repository and trimesh patch

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

### 1.3 Conda environments

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

### 1.4 Verify the installation

After creating the data ([section 2](#2-synthetic-data)):

```bash
source env_setup.sh            # activates ricostruzione3d (the WAYLAND fix only matters on WSLg)

# one terrain per family, one method at a time: a failure points to the environment at fault
python evaluate.py --split val --limit 1 --methods poisson
python evaluate.py --split val --limit 1 --methods ppsurf
```

Each command creates `output/evaluation/<date_time>_val/`. Check that
`metrics_per_terrain.csv` has 6 rows with an empty `error` column,
and that `meshes/` contains 6 `.ply` files. Poisson takes a few seconds per
terrain. PPSurf takes 38–43 s per terrain on a GTX TITAN X with
`rec_batch_size=50000`, and longer with the default 10000; the first run also
downloads the weights.

## 2. Synthetic data

In the `ricostruzione3d` environment:

```bash
python utils/generate_seabed_heightmaps.py --with-structures --n-per-family 17
python utils/simulate_multibeam_scans.py
```

- `generate_seabed_heightmaps.py` writes 100 × 100 m heightmaps in 6 families
  (`flat_sand`, `sand_ripples`, `ridges`, `rocky`, `slope`, `mixed`) with pipes
  and pillars, split per terrain. `--n-per-family 17` with the default seed
  (`--seed 0`) reproduces the current dataset. It ends with:
  ```
  Generated 102 terrains -> data/synthetic  (train 72, val 18, test 12)
  ```
- `simulate_multibeam_scans.py` reads `data/synthetic/manifest.csv` and writes,
  in each `data/synthetic/<split>/<id>/` folder:
  - `<id>_scan.csv`: the point cloud, columns
    `x,y,z,intensity,ping_number,elapsed_s` (Y is vertical, `elapsed_s` is the
    time since the start of the scan);
  - `<id>_ground_truth.obj`: the true mesh, closed (seabed + walls + bottom);
  - `<id>_scan.json`: scan parameters and applied offsets.

  Its parameters (sensor, survey path, which terrains) are constants at the top
  of the script: 200 beams per ping, 120° aperture, 6 pings/s, 1 m/s,
  lawn-mower path with 30% line overlap.

The evaluation uses the `test+val` split, the union of `test` and `val`:

```bash
mkdir -p data/synthetic/test+val
cp -r data/synthetic/test/* data/synthetic/val/* data/synthetic/test+val/
```

Check the folder sizes:

```bash
for s in train val test test+val; do echo "$s: $(ls data/synthetic/$s | wc -l)"; done
# expected: train: 72, val: 18, test: 12, test+val: 30
```

## 3. Usage

Run everything from the repository root, with the environment active
(`source env_setup.sh`).

**Evaluation.** The four configurations of the report:

```bash
python evaluate.py --methods poisson --split test+val
python evaluate.py --methods ppsurf  --split test+val --set ppsurf.rec_batch_size=50000
python evaluate.py --methods poisson --split test+val --sliding_window on
python evaluate.py --methods ppsurf  --split test+val --sliding_window on --set ppsurf.rec_batch_size=50000
```

| Option | Meaning |
|---|---|
| `--split {train,val,test,test+val}` | data split; default from `evaluation.split` (`test+val`) |
| `--methods poisson ppsurf` | methods to run, in sequence on each terrain; default both |
| `--limit N` | the first N terrains **of each family** (alphabetical order), not N in total |
| `--sliding_window {on,off}` | overrides `sliding_window.enabled` |
| `--set SECTION.KEY=VALUE ...` | overrides any config value for this run, e.g. `--set ppsurf.gen_refine_iter=5 poisson.depth=8`; repeatable |
| `--tau T` | threshold in metres (default `evaluation.tau`, 0.5); several values only with `--recompute` |
| `--recompute RUN_FOLDER` | no reconstruction; recomputes Chamfer, precision, recall and F-score for each `--tau` from the raw distances of an existing run |

Without arguments, `evaluate.py` uses the defaults of the `evaluation` section
of the config (`test+val`, all terrains, Poisson and PPSurf, full scan).

**Metrics at other thresholds**, without reconstructing:

```bash
python evaluate.py --recompute output/evaluation/<run> --tau 0.1 0.2 0.5
```

**Screenshots** of every mesh of one or more runs, with the same camera for
the same terrain across runs (written to `<run>/screenshots/`), and of the
ground truth (written to `output/groundtruth/`):

```bash
python visualization/screenshot_run.py output/evaluation/<run1> output/evaluation/<run2>
python visualization/screenshot_run.py --ground_truth --data data/synthetic/test+val
```

**Interactive viewer** for one mesh (the path is set in `MESH_PATH` at the
top of the file):

```bash
python visualization/visualize_mesh.py
```

## Contacts

Simone Dario ([simone.dario@univr.it](mailto:simone.dario@univr.it)), Università degli Studi di Verona
