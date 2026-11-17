# Analysis: experiments, parameter choices and discarded methods

Companion to the [README](README.md): this file explains *why* the pipeline is set up the way it is. The
README covers installation and usage; the method and final results are in the report,
[report.pdf](report.pdf).

## Contents

1. [Window size and PPSurf holes](#window-size-and-ppsurf-holes)
2. [The benchmark folder: experiments and results](#the-benchmark-folder-experiments-and-results)
3. [Why PPSurf is configured this way](#why-ppsurf-is-configured-this-way)
4. [Discarded methods](#discarded-methods)

## Window size and PPSurf holes

With 300-ping windows (~50 m, laptop run, `PPSURF_2026-11-10_23-40-13_test+val_sw`, 2 terrains per family)
PPSurf + SW left many holes on the smooth seabeds. Halving the window removes them:

| Recall τ=0.5 (PPSurf + SW) | flat_sand | mixed | ridges | rocky | sand_ripples | slope |
|---|---|---|---|---|---|---|
| 300-ping windows | 0.799 | 0.946 | 0.919 | 0.986 | 0.845 | 0.858 |
| 150-ping windows | **0.996** | **0.998** | **0.997** | **0.999** | **0.996** | **0.984** |

Other tests on `flat_sand_012` and the other `*_012` terrains before choosing:

- **Grid 193 instead of 129** (300-ping windows): recall +2.5/+5 points on `mixed` and `ridges`, time ×2.3,
  and `flat_sand_012` timed out. The marching-cubes cell was not the main cause.
- **Overlap 30 instead of 60** (300-ping windows): no effect (recall 0.839 → 0.843).
- **150-ping windows**: recall at τ=0.2 0.70 → 0.98, p95 seabed→mesh 1.22 → 0.16 m. A few small holes
  remain in the screenshots, ~2% of the surface.

So the PPSurf limit depends on the extent of the scene it receives, not on the grid: the same effect seen
with the 25 m squares in the [benchmark (test 5)](#5-tiled-ppsurf-tile_ppsurfpy).

## The benchmark folder: experiments and results

The scripts in `benchmark/` are targeted experiments that answer specific questions about cost and quality.
They all run from the repository root and write to `benchmark/benchmark_output/`, which git ignores: the
results are reported below. Unless stated otherwise, the metrics are those of `evaluate.py` (against the
seabed, τ = 0.5 m).

| Script | Question | Output |
|---|---|---|
| `time_poisson.sh`, `time_ppsurf.sh` | How much does each phase of each method cost? | `timing.csv`, `poisson/`, `ppsurf/` |
| `poisson_density_sweep.py` | What is the right threshold for the Poisson density trim? | `poisson_density_sweep.csv` |
| `single_layer_ppsurf.py` | How much does the PPSurf double shell weigh? | `single_layer_ppsurf.csv`, `ppsurf_single_layer/` |
| `crop_ppsurf.py` | Is the PPSurf limit the grid or the model? | `crop_ppsurf*.csv`, `ppsurf_crop/` |
| `tile_ppsurf.py` | Does tiled PPSurf reach Poisson? | `tile_ppsurf.csv`, `ppsurf_tile/` |
| `common.sh`, `prepare_input.py`, `poisson_phases.py`, `surface_metrics.py` | helpers for the timing scripts | – |

### 1. Time per phase (`time_*.sh`)

```bash
bash benchmark/time_poisson.sh data/synthetic/val/flat_sand_012/flat_sand_012_scan.csv 3
bash benchmark/time_ppsurf.sh  data/synthetic/val/flat_sand_012/flat_sand_012_scan.csv 3
```

`prepare_input.py` runs the preprocessing once, outside the timer. Then each phase of the method is timed
with its peak RAM and VRAM. For PPSurf, `library_startup` measures only the fixed start-up cost (Python,
PyTorch, CUDA) and `rec` the full reconstruction. On `flat_sand_012` (77 702 points after preprocessing),
GTX 1650 with 4 GB, 3 repetitions:

| Method | Phase | Time | Peak RAM | Peak VRAM |
|---|---|---|---|---|
| Poisson | reconstruction (octree depth 9) | 5–12 s | ~470 MB | – |
| Poisson | full process (start-up + normals + reconstruction + cleanup) | 9–16 s | ~470 MB | – |
| PPSurf | library start-up + CUDA | 4–10 s | ~690 MB | – |
| PPSurf | full `pps.py rec` | ~170 s | ~1.6 GB | ~1.9 GB |

PPSurf costs about 10–20 times Poisson, and its time depends on the marching-cubes grid, not on the number of
points. The timing CSV also holds Chamfer and F-score of these runs, but they predate the Poisson normals
fix: for quality, the tables below apply.

### 2. Poisson density-trim threshold (`poisson_density_sweep.py`)

Poisson runs once per terrain. The raw mesh is then trimmed at several density quantiles and each is
measured. Test on 6 val terrains, 1 per family:

| Quantile | 0.0 | 0.02 | 0.03 | **0.05** | 0.07 | 0.1 | 0.2 | 0.35 |
|---|---|---|---|---|---|---|---|---|
| Mean F-score | 0.903 | 0.976 | 0.990 | **0.998** | 0.995 | 0.984 | 0.936 | 0.820 |
| Mean Chamfer | 0.34 m | 0.08 m | 0.06 m | **0.05 m** | 0.05 m | 0.07 m | 0.13 m | 0.30 m |

The maximum at 0.05 is clear and is the value in the config. Without trimming, spurious "curtains" remain
at the borders; trimming too much opens holes.

Before this sweep the normals were estimated with a fixed 0.1 m radius. After the 0.5 m voxel no point had
neighbours within that radius, so all normals stayed (0,0,1). Now `estimate_normals` uses the 30 nearest
neighbours.

### 3. The PPSurf double shell (`single_layer_ppsurf.py`)

PPSurf, pretrained on closed objects, wraps the open seabed in a thin solid: almost every vertical ray
crosses **two** surfaces, one above and one below the true seabed, about 1 m apart. The script works on the
meshes already saved by `evaluate.py`, without a GPU. On a 0.25 m XZ grid it takes the first hit from above
(top layer), the first from below (bottom layer) and their midpoint. Mean over the 6 `*_012` terrains:

| Variant | Chamfer | Precision | Recall | F-score |
|---|---|---|---|---|
| original PPSurf | 0.344 m | 0.67 | 0.96 | 0.79 |
| top layer | 0.548 m | 0.52 | 0.52 | 0.52 |
| bottom layer | 0.374 m | 0.80 | 0.75 | 0.77 |
| middle layer | 0.262 m | 0.91 | 0.86 | 0.88 |
| Poisson | 0.048 m | 1.00 | 1.00 | 1.00 |

The shell explains part of the error, not all of it: even the midpoint is more than 5 times worse than
Poisson. Using the input without downsampling (250k points) changes nothing: Chamfer 0.42 against 0.46 m on
`flat_sand_012`.

### 4. Grid or model? (`crop_ppsurf.py`)

With `gen_resolution_global: 129` on a 100 m terrain the marching-cubes cells are ~0.8 m, against ~0.2 m for
Poisson. The script reruns PPSurf on a 25×25 m crop, where the same grid gives ~0.2 m cells, then measures
all variants on the inner 20 m square. Chamfer in metres:

| Terrain | Full | Full, middle layer | Crop | **Crop, middle layer** | Poisson |
|---|---|---|---|---|---|
| flat_sand_012 | 0.454 | 0.301 | 0.113 | **0.035** | 0.039 |
| ridges_012 | 0.302 | 0.325 | 0.107 | **0.054** | 0.031 |
| rocky_012 | 0.292 | 0.326 | 0.111 | **0.093** | 0.047 |
| **Mean** | 0.349 | 0.317 | 0.110 | **0.061** | 0.039 |

On the crop the shell drops from 0.9–1.2 m to 0.25–0.34 m, and with the middle layer PPSurf approaches Poisson.
But the crop reduces both the cell and the scene extent, so this test alone does not say which one matters.

**The test that separates the two effects**: a finer grid on the whole terrain (GTX TITAN X with 12 GB, val
split, 1 terrain per family):

| `gen_resolution_global` | Cell over 100 m | Chamfer | F-score | Time/terrain |
|---|---|---|---|---|
| 129 | ~0.8 m | 0.360 m | 0.779 | ~46 s |
| 257 (paper default) | ~0.4 m | 0.363 m | 0.782 | ~187 s |
| 519 | ~0.2 m | 0.354 m | 0.786 | ~918 s |

Resolution ×4, time ×20, no improvement: the differences are within the run-to-run noise (~0.015 m). At 519
the shell stays at 0.8–0.9 m. **The shell thickness is about 1% of the scene side** (100 m → ~0.9 m,
25 m → ~0.3 m), whatever the grid. This is consistent with PPSurf normalising each point cloud into the unit
cube: the shell has a fixed thickness in normalised coordinates. The limit is in the field predicted by the
network, not in the discretisation.

### 5. Tiled PPSurf (`tile_ppsurf.py`)

Applies the previous conclusion to the whole terrain. The terrain is split into 25 m squares (20 m core +
2.5 m margin per side, discarded when merging). All squares go through a single PPSurf subprocess (dataset
mode, model loaded once) and are merged in two ways: `tile_original` (meshes cropped to the cores) and
`tile_middle` (middle layer on a global grid). Grid 129 per square, 25 squares per terrain, `*_012`
terrains, TITAN X:

| Variant | Chamfer | Precision | Recall | F-score | Shell thickness |
|---|---|---|---|---|---|
| PPSurf on the whole terrain | 0.344 m | 0.67 | 0.96 | 0.79 | ~1.0 m |
| PPSurf on the whole terrain, middle layer | 0.262 m | 0.91 | 0.86 | 0.88 | ~1.0 m |
| **tiled, original** | **0.121 m** | 0.98 | 0.99 | **0.99** | 0.28 m |
| **tiled, middle layer** | **0.086 m** | 0.99 | 0.99 | **0.99** | 0.28 m |
| Poisson | 0.048 m | 1.00 | 1.00 | 1.00 | – |

At τ = 0.5 m tiled PPSurf is on par with Poisson. Its Chamfer is still about 2.5 times Poisson's (1.8 with the
middle layer), but the whole gap lies below the threshold. The middle layer adds little and, being 2.5D, erases pipes and pillars:
`tile_original` is the variant to keep.

**Cost remains the obstacle**: ~17 min per terrain (~41 s per square, set by the grid) against ~7 s for
Poisson, about 150 times as much.

## Why PPSurf is configured this way

- **Why PPSurf and not another neural method.** It is patch-based: it reconstructs piece by piece instead of
  deforming a global shape, so it suits open, elongated surfaces such as a multibeam scan. The other
  candidates were discarded (see [Discarded methods](#discarded-methods)). SA-ConvONet, the last one left,
  was worse than PPSurf (F-score ~0.59 against ~0.70 on `flat_sand_012`). Its per-scene adaptation variant
  was infeasible: ~1.5 h per terrain, and OOM on 4 GB.
- **`gen_resolution_global: 129`** (default 257): on 4 GB of VRAM the default runs out of memory. On the
  12 GB TITAN X, 257 and 519 do not improve quality (test 4) and cost 4 and 20 times more. There is no reason
  to raise it.
- **`rec_batch_size: 10000`** (default 50000): only needed to fit in 4 GB during marching cubes. It does not
  change the result. On the TITAN X 50000 is used (`--set`), with ~8.6 GB of VRAM.
- **`gen_subsample_manifold_iter: 3` and `gen_refine_iter: 3`** (defaults 10 and 10): about a third less
  time, with the same or slightly better quality.
  - The first parameter is *test-time augmentation*: the global part of the network is repeated on subsets
    of the points, until each point has been used as many times as the value set.
  - The second is the number of iterations PPSurf uses to refine the marching-cubes intersections.

  Tests on the 6 `*_012` terrains, without SW, GTX 1650 (with `evaluate.py --set`):

  | TTA / refine | Time per terrain | F τ=0.1 | F τ=0.2 | F τ=0.5 |
  |---|---|---|---|---|
  | 10 / 10 (default) | 176 s | 0.151 | 0.315 | 0.801 |
  | 5 / 10 | 167 s | – | – | 0.808 |
  | 3 / 10 | 164 s | – | – | 0.818 |
  | 10 / 5 | 133 s | 0.151 | 0.314 | 0.806 |
  | 10 / 3 | 119 s | 0.149 | 0.313 | 0.801 |
  | **3 / 3** | **112 s** | **0.169** | **0.351** | **0.817** |

  The real lever is refinement: placing the surface better inside the cell is pointless when the surface is
  shifted by 20–50 cm anyway because of the double shell. Test-time augmentation weighs little on time, and
  reducing it improved quality slightly in both cases. A hypothesis: with less averaging the shell is less
  regular and touches the true seabed more often. It is not verified, and the test covers only 6 terrains.

  The remaining ~110 s are the cost of network inference on the grid points near the surface. PPSurf does
  not evaluate the whole cube: it uses a *region-growing* marching cubes that starts from the input points
  (paper, Sec. 3.3).
- **Same preprocessing as Poisson** (0.5 m voxel + outliers): removing the downsampling does not change
  quality and lengthens the run (test 3). `max_points: 250000` is only a safety limit.
- **No middle layer in the pipeline**: it improves the Chamfer a little but is 2.5D, so it erases exactly the
  structures (pipes, pillars) where a neural method could beat Poisson.
- **Small scenes rather than fine grids**: since the shell scales with the scene extent (tests 4 and 5), the
  right lever is to reconstruct smaller scenes. Tiling works but costs too much. The sliding window produces
  small scenes by construction and inherits the benefit: with 150-ping windows F at τ=0.2 goes from 0.14–0.54
  to 0.80–0.86 (see the [report](report.pdf) and [Window size](#window-size-and-ppsurf-holes)).
- **Possible next steps**: keep the model loaded between sliding-window windows (5–10 s saved per window), or
  fine-tune on open surfaces, which attacks the cause of the shell. The authors themselves call the
  reconstruction times *"non-interactive"* (paper, Sec. 4.3).

## Discarded methods

| Method | Reason |
|---|---|
| Point2Mesh | Starts from a convex hull, meant for compact genus-0 shapes. On the scan strip more iterations worsen fidelity. 1000 s for a result 3.5 times worse than Poisson. |
| SA-ConvONet | Worse than PPSurf on the synthetic data. The per-scene optimisation (the "SA" part) takes ~1.5 h per terrain and runs out of memory on 4 GB. Dependencies stuck at PyTorch 1.4 / CUDA 10.1. Removed from the code, recoverable from the git history. |
| NKSR | Wheels no longer available, must be built with CUDA 12.8; likely OOM on 4 GB. |
| DeepSDF, IF-Net | Need training on a shape dataset (shape prior); the available weights are on ShapeNet. |
| NeuS, 3DGS | Need multi-view RGB images, while the project stays on sonar. |
| PointNet++, DGCNN | Feature extractors, they do not produce meshes. |
| Diffusion methods (WALA, Make-A-Shape) | Immature on sparse, noisy input; trained on objects, not scenes. |
