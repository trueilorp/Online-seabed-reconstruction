# Seabed 3D reconstruction from multibeam sonar

This repository reconstructs a 3D mesh of the seabed from multibeam sonar point
clouds and evaluates it against a known ground truth. It compares
**Poisson** surface reconstruction (Open3D) and **PPSurf** (neural,
pretrained), each on the full scan and with a **sliding window** (SW) that
processes the scan in windows of pings, as it would arrive during a survey.
An online version of the Poisson pipeline reconstructs the seabed while a Unity
simulation acquires it over a real seabed of the Venice Lagoon (section 4).

This README covers installation and usage. Method, results and diagnostic
experiments are in the project report, [report.pdf](report.pdf);
the experiment notes are in [ANALYSIS.md](ANALYSIS.md).

## Contents

1. [Installation](#1-installation)
2. [Synthetic data](#2-synthetic-data)
3. [Usage](#3-usage)
4. [Online reconstruction from Unity](#4-online-reconstruction-from-unity)
5. [Contacts](#contacts)

## 1. Installation

The git repository contains the code and the real-seabed patch of section 4
(`data/real/patch_malamocco/`); the neural repository (`neural_methods/`), the
synthetic data (`data/synthetic/`) and the results (`output/`) are excluded and
must be recreated. Run every command from the repository root.

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
pip install psutil          # optional: without it, the RAM peak (peak_ram_mb) is not recorded

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

## 4. Online reconstruction from Unity

<!-- TODO: link to the Unity project -->
A Unity simulation drives a boat over a real seabed (the Malamocco patch of the Venice Lagoon) and records the
sensor returns (a VLP-16 used as a proxy of a multibeam sonar) to a CSV while it runs. The mesh of the seabed is
reconstructed with Poisson **while the CSV grows**, then evaluated against the real bathymetry and rendered
as a video.

### 4.1 Ground truth and Unity scene

The patch used in the experiments is already in the repository, in `data/real/patch_malamocco/` (256 × 256 m,
1 m cells): the Unity heightmap (`heightmap_257.raw`), the ground-truth mesh (`gt_mesh.ply`), the mask of the
cells with real data (`valid_mask.npy`) and the Terrain parameters (`meta.json`).

To cut another patch, download the lagoon DTM (MGDS, [doi:10.1594/IEDA/323605](https://doi.org/10.1594/IEDA/323605),
ESRI ASCII; the 421 MB `iim_malamocco.asc` is not in the repository) into `data/real/` and run
`data/real/venice_to_unity.py` (without `--center` it picks the patch with the most valid cells and relief):

```bash
python data/real/venice_to_unity.py data/real/iim_malamocco.asc --out data/real/patch_malamocco --size 256
```

Frames:

| Data | Frame |
|---|---|
| `gt_mesh.ply` | local ENU: x east, y north, z up = absolute DTM elevation; origin at the SW corner of the patch |
| Unity CSV and every online mesh | Unity world: X, **Y up**, Z north |


### 4.2 Reconstruction while Unity runs

Launch it before or after putting the scene in Play: with `--latest` it waits until a CSV modified after the
launch appears in the folder; with `--csv` it waits for
that file to exist.

```bash
python online_reconstruction.py --latest "/mnt/c/Users/<user>/.../Recordings"          # waits for the new CSV
python online_reconstruction.py --latest "/mnt/c/Users/<user>/.../Recordings" --view   # also a live 3D window
python online_reconstruction.py --csv <scan.csv> --replay --speed 4                    # replay a finished recording
```

Pipeline of each update:

1. **Reading.** The CSV (`x,y,z,intensity,ping_number,elapsed_s`, ~1 ping/s, ~21k points per ping) is tailed:
   only the newly appended complete lines are read, never the whole file again.
2. **Batching.** Rows are grouped into batches of `online.update_every_pings` complete pings (default 5); a ping
   is complete when the next one starts.
3. **Voxel map.** The batch is added to a map of 0.5 m voxels (`preprocessing.downsampling.voxel_size`) that keeps
   the running centroid of each voxel; repeated passes over the same area are averaged.
4. **Region.** The plan-view (XZ) box covered by the batch, plus `online.margin` (2.5 m), selects the map voxels
   given to Poisson (including older data in that area).
5. **Poisson**, same steps as the offline pipeline: statistical outlier removal (20 neighbours, 2σ), normals on the
   30 nearest neighbours oriented up (+Y), Poisson depth 9 on `online.poisson_threads` threads, removal of the 5%
   lowest-density vertices, mesh cleanup, removal of the triangles farther than 1 m from the points.
6. **Merge.** The rebuilt region (cut to the box) replaces the triangles of the merged mesh inside the box.
7. **Write.** `mesh.ply` is rewritten atomically (a reader never sees half a file); a copy goes to `meshes/`.

The run stops after `online.idle_timeout_s` (15 s) without new data, or with Ctrl+C. Output in
`output/online/<date_time>/`:

| File | Content |
|---|---|
| `mesh.ply` | latest merged mesh (Unity frame) |
| `meshes/mesh_<update>.ply` | the mesh after each update (`--save-every`, `online.snapshot_every`) |
| `updates.csv` | one row per update: pings, `elapsed_s` (simulation time of the last point), voxels, box, triangles, `latency_s` (batch complete → mesh written), `budget_s`, `rtf` = latency / budget, time of each phase (`t_read` … `t_snapshot`), `arrived_unix_s` / `ready_unix_s` (wall clock) |
| `config.txt`, `log.txt` | CSV path, effective config, console output |

### 4.3 Metrics against the ground truth

```bash
python evaluate_online.py output/online/<run> --tau 0.2 0.3 0.4 0.5
```

Moves the ground truth into the Unity frame (`--terrain-position`, default −128 −128), keeps the cells with real
DTM data and, for each snapshot, the area seen by the sensor up to that time (cells within `--coverage-radius`,
1 m, of a CSV point). Chamfer, precision, recall and F-score at each τ are computed as in `evaluate.py` and written
to `<run>/metrics_online.csv` (one row per snapshot and τ); `gt_covered.ply` is the ground truth of the last
snapshot, to check the alignment visually. A median reconstruction→GT distance above 2 m is flagged as a
probable misalignment.

### 4.4 Report figures and tables

```bash
python visualization/plot_online.py output/online/<run>
```

Writes to `imgs/` the figures of the report section on the online reconstruction (`online_quality.png`: F-score
and Chamfer per update; `online_timing.png`: latency per update split by phase; `online_plan.png`: plan view of
the area seen by the sensor, covered by the mesh and left uncovered; `online_mesh.png`: screenshot of the final
mesh, vertical exaggeration `--zexag`, default 2), and to `report/tables/` the summary table
(`online.tex`) and the numbers quoted in the text (`online_values.tex`, LaTeX macros), so figures, table and text
always come from the same run. Needs `metrics_online.csv` with τ = 0.2 and 0.5 m (section 4.3).

### 4.5 Video and GIF

```bash
python video/render_video.py --run output/online/<run> --unity-video <Recordings/Movie_XXX.mp4> --preview
python video/render_video.py --run output/online/<run> --unity-video <Recordings/Movie_XXX.mp4>
python visualization/online_gif.py output/online/<run>
```

`render_video.py` writes a 1920×1080 H.264 video to `output/video/video_<date_time>.mp4` (`--out` for another
path); with `--preview` it writes only three frames (10%, 50%, 95% of the session), the title card and the
summary to `output/video/preview_<run>/`. The CSV is read from `<run>/config.txt` (`--csv` to override) and
`metrics_online.csv` must contain τ = 0.2 and 0.5 m. Layout, on a common clock (`elapsed_s`, simulation time):

- top: the Unity recording (`--unity-video`, centre-cropped), the point cloud acquired so far (coloured by
  depth for the last `online.update_every_pings` pings, grey before, the newest ping in white) and the online
  mesh, with the same camera fitted on the final mesh and vertical exaggeration `--zexag` (default 2);
- bottom: a plan view (GT hillshade, area covered by the mesh, boat track as the centre of each ping
  footprint, box rebuilt at the last update) and F-score, Chamfer and latency per update with a time cursor;
- a title card at the start and a summary table at the end.

The mesh shown at time t is the latest one already **ready** at t. The wall-clock timestamps of `updates.csv`
are mapped to the simulation clock (Unity can run slower than real time), so the mesh delay shown is on the same
clock as the video. The values behind the plots and the summary are written to `<run>/metrics_per_update.csv`.
`online_gif.py` makes a GIF of the snapshots alone (`<run>/online.gif`).

## Contacts

Simone Dario ([simone.dario@univr.it](mailto:simone.dario@univr.it)), Università degli Studi di Verona
