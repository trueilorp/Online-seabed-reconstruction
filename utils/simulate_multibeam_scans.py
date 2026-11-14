"""
Simulates a multibeam survey over the synthetic terrains of generate_seabed_heightmaps.py and writes, per
terrain, <id>_scan.csv (x,y,z,intensity,ping_number,elapsed_s; Y vertical), <id>_ground_truth.obj (closed
terrain mesh, same coordinates) and <id>_scan.json (scan parameters and offsets).

Simplified sensor model (not acoustic physics): lawn-mower path at constant depth, a fan of N_BEAMS rays per
ping perpendicular to the track with random roll/pitch, exact raycasts on the full-resolution terrain, range
noise, dropout and water-column outliers. Defaults estimated from OS3D_Max_Range_Test.csv (rate, speed,
aperture), not from the datasheet; intensity is a proxy. To imitate the Unity sensor: MIN_RANGE=1,
MAX_RANGE=50, RANGE_SIGMA=0.05, DROPOUT=0, OUTLIER_RATE=0.

Usage: python utils/simulate_multibeam_scans.py   (parameters are the constants below)
"""
import csv
import json
import os
import time

import numpy as np
import open3d as o3d

# --- input / terrain selection
DATASET_DIR = "data/synthetic"   # folder with manifest.csv (output of generate_seabed_heightmaps.py)
ONLY = None                      # ids to process, e.g. ["rocky_000"]; None = all
SPLITS = None                    # e.g. ["train", "val"]; None = all splits
LIMIT = None                     # maximum number of terrains; None = no limit
SEED = 0                         # global scan seed (with the terrain seed it determines everything)

# --- sensor
FAN_DEG = 120.0                  # total fan aperture, degrees
N_BEAMS = 200                    # beams (points) per ping
PING_RATE = 6.0                  # pings per second
MIN_RANGE = 0.5                  # m
MAX_RANGE = 50.0                 # m
RANGE_SIGMA = 0.15               # Gaussian range noise, m
DROPOUT = 0.02                   # probability that a ray returns nothing
OUTLIER_RATE = 0.005             # fraction of spurious echoes in the water column (always above the seabed)
ROLL_SIGMA_DEG = 0.3             # roll jitter per ping, degrees
PITCH_SIGMA_DEG = 0.2            # pitch jitter per ping, degrees
INTENSITY_SCALE = 4.0            # proxy: intensity = scale * cos(incidence angle)

# --- survey
SPEED = 1.0                      # platform speed, m/s (along-track spacing = SPEED/PING_RATE)
OVERLAP = 0.3                    # overlap between adjacent lines (0-1)
MAX_LINES = 40                   # maximum number of lines
SENSOR_Y = -0.5                  # sensor Y (like the sensor under the hull in Unity)
MEAN_DEPTH = 8.0                 # mean seabed depth, m below Y=0
CLEARANCE = 2.0                  # minimum distance between sensor and terrain top, m

# --- ground truth
GT_STEP = 2                      # GT grid subsampling (1 = full resolution; must divide resolution-1)
CLOSED_GT = True                 # True = closed mesh with walls and bottom (for occupancy methods)
BASE_THICKNESS = 1.0             # base thickness if CLOSED_GT, m

CSV_HEADER = "x,y,z,intensity,ping_number,elapsed_s"


def load_heightmap(path: str, res: int, height_range: float) -> np.ndarray:
    raw = np.fromfile(path, dtype="<u2")
    if raw.size != res * res:
        raise ValueError(f"{path}: expected {res * res} values, found {raw.size}")
    return raw.reshape(res, res).astype(np.float64) / 65535.0 * height_range


def grid_vertices(h: np.ndarray, size: float, y_shift: float, step: int = 1) -> np.ndarray:
    res = h.shape[0]
    xs = np.linspace(-size / 2, size / 2, res)[::step]
    X, Z = np.meshgrid(xs, xs)  # X[r, c] = xs[c], Z[r, c] = xs[r]  (row <-> Z, column <-> X)
    Y = h[::step, ::step] + y_shift
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def grid_faces(n: int) -> np.ndarray:
    """Triangles of an n x n grid, with normals towards +Y."""
    r, c = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (r * n + c).ravel()
    b, cc, d = a + 1, a + n, a + n + 1
    return np.concatenate([np.stack([a, cc, b], 1), np.stack([b, cc, d], 1)]).astype(np.int64)


def close_mesh(verts: np.ndarray, faces: np.ndarray, n: int, thickness: float):
    """Adds side walls and a bottom (fan from the centre): closed, oriented mesh."""
    loop = np.concatenate([
        np.arange(0, n),                           # row 0, increasing columns
        np.arange(1, n) * n + (n - 1),             # last column, increasing rows
        (n - 1) * n + np.arange(n - 2, -1, -1),    # last row, decreasing columns
        np.arange(n - 2, 0, -1) * n,               # first column, decreasing rows
    ])
    L = len(loop)
    y_bot = verts[:, 1].min() - thickness
    bot = verts[loop].copy()
    bot[:, 1] = y_bot
    center = np.array([[verts[:, 0].mean(), y_bot, verts[:, 2].mean()]])
    n0 = len(verts)
    verts2 = np.concatenate([verts, bot, center])
    m = n0 + L
    new = []
    for i in range(L):
        j = (i + 1) % L
        p, q = loop[i], loop[j]
        pb, qb = n0 + i, n0 + j
        new += [(p, q, qb), (p, qb, pb), (m, pb, qb)]
    return verts2, np.concatenate([faces, np.array(new, dtype=np.int64)])


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray, comment: str):
    with open(path, "w") as f:
        f.write(f"# {comment}\n")
        np.savetxt(f, verts, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def make_scene(verts: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts.astype(np.float32), dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32))
    return scene


def choose_y_shift(h: np.ndarray) -> float:
    """Mean seabed depth = MEAN_DEPTH, but the terrain top stays at least CLEARANCE metres below the
    sensor (otherwise the sensor would end up inside the rock)."""
    shift = -MEAN_DEPTH - h.mean()
    top_limit = SENSOR_Y - CLEARANCE
    if h.max() + shift > top_limit:
        shift = top_limit - h.max()
    return float(shift)


def plan_survey(size: float, alt_mean: float):
    swath = 2.0 * alt_mean * np.tan(np.deg2rad(FAN_DEG / 2.0))
    spacing = max(swath * (1.0 - OVERLAP), 1e-3)
    n_lines = int(np.clip(np.ceil(size / spacing), 1, MAX_LINES))
    z_lines = (np.arange(n_lines) + 0.5) * size / n_lines - size / 2
    ping_dx = SPEED / PING_RATE
    n_pings = int(np.floor(size / ping_dx)) + 1
    x_pings = -size / 2 + np.arange(n_pings) * ping_dx
    return z_lines, x_pings, swath


def simulate(scene, z_lines, x_pings, rng) -> np.ndarray:
    nb = N_BEAMS
    theta = np.deg2rad(np.linspace(-FAN_DEG / 2, FAN_DEG / 2, nb))
    chunks = []
    ping_counter = 0
    for li, zl in enumerate(z_lines):
        xs = x_pings if li % 2 == 0 else x_pings[::-1]  # lawn-mower
        npg = len(xs)
        roll = np.deg2rad(rng.normal(0.0, ROLL_SIGMA_DEG, npg))
        pitch = np.deg2rad(rng.normal(0.0, PITCH_SIGMA_DEG, npg))
        th = theta[None, :] + roll[:, None]
        d = np.stack([np.broadcast_to(np.sin(pitch)[:, None], th.shape),
                      -np.cos(th) * np.cos(pitch)[:, None],
                      np.sin(th)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        o = np.empty_like(d)
        o[..., 0] = xs[:, None]
        o[..., 1] = SENSOR_Y
        o[..., 2] = zl
        rays = np.concatenate([o, d], axis=-1).reshape(-1, 6).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t = ans["t_hit"].numpy().astype(np.float64)
        nrm = ans["primitive_normals"].numpy().astype(np.float64)
        dd = d.reshape(-1, 3)
        oo = o.reshape(-1, 3)

        hit = np.isfinite(t)
        cos_inc = np.clip(np.abs(np.sum(nrm * dd, axis=1)), 0.05, 1.0)
        cos_inc[~hit] = 0.05
        t_noisy = t + rng.normal(0.0, RANGE_SIGMA, t.shape)
        valid = hit & (rng.random(t.shape) >= DROPOUT)

        # water-column echoes: only on rays that hit the seabed, and always before it
        out_mask = (rng.random(t.shape) < OUTLIER_RATE) & hit
        t_out = rng.uniform(MIN_RANGE, np.maximum(np.where(hit, t, MAX_RANGE), MIN_RANGE + 1e-3))
        t_use = np.where(out_mask, t_out, t_noisy)
        valid = (valid | out_mask) & (t_use >= MIN_RANGE) & (t_use <= MAX_RANGE)
        cos_inc = np.where(out_mask, 0.05, cos_inc)

        pts = oo + dd * t_use[:, None]
        ping_id = ping_counter + np.repeat(np.arange(npg), nb)
        chunk = np.column_stack([pts, INTENSITY_SCALE * cos_inc, ping_id, ping_id / PING_RATE])
        chunks.append(chunk[valid])
        ping_counter += npg
    return np.concatenate(chunks)


def write_csv(path: str, arr: np.ndarray):
    np.savetxt(path, arr, delimiter=",", header=CSV_HEADER, comments="",
               fmt=["%.5f", "%.5f", "%.5f", "%.4f", "%d", "%.6f"])


def main():
    with open(os.path.join(DATASET_DIR, "manifest.csv")) as f:
        rows = list(csv.DictReader(f))
    if ONLY:
        rows = [r for r in rows if r["id"] in ONLY]
    if SPLITS:
        rows = [r for r in rows if r["split"] in SPLITS]
    if LIMIT:
        rows = rows[:LIMIT]
    if not rows:
        raise SystemExit("no terrain selected")

    sensor_params = dict(fan_deg=FAN_DEG, n_beams=N_BEAMS, ping_rate=PING_RATE, min_range=MIN_RANGE,
                         max_range=MAX_RANGE, range_sigma=RANGE_SIGMA, dropout=DROPOUT,
                         outlier_rate=OUTLIER_RATE, roll_sigma_deg=ROLL_SIGMA_DEG,
                         pitch_sigma_deg=PITCH_SIGMA_DEG, sensor_y=SENSOR_Y, speed=SPEED, overlap=OVERLAP)

    for r in rows:
        t0 = time.time()
        res = int(r["resolution"])
        size = float(r["size_m"])
        hr = float(r["height_range_m"])
        if (res - 1) % GT_STEP != 0:
            raise SystemExit(f"GT_STEP={GT_STEP} does not divide resolution-1 = {res - 1}")
        d = os.path.join(DATASET_DIR, os.path.dirname(r["raw"]))
        h = load_heightmap(os.path.join(DATASET_DIR, r["raw"]), res, hr)
        y_shift = choose_y_shift(h)

        verts = grid_vertices(h, size, y_shift)
        scene = make_scene(verts, grid_faces(res))

        z_lines, x_pings, swath = plan_survey(size, SENSOR_Y - (h.mean() + y_shift))
        rng = np.random.default_rng(np.random.SeedSequence([int(r["seed"]), SEED, 1]))
        pts = simulate(scene, z_lines, x_pings, rng)
        write_csv(os.path.join(d, f"{r['id']}_scan.csv"), pts)

        n_gt = (res - 1) // GT_STEP + 1
        vg = grid_vertices(h, size, y_shift, GT_STEP)
        fg = grid_faces(n_gt)
        if CLOSED_GT:
            vg, fg = close_mesh(vg, fg, n_gt, BASE_THICKNESS)
        write_obj(os.path.join(d, f"{r['id']}_ground_truth.obj"), vg, fg,
                  f"ground truth {r['id']} (grid {n_gt}x{n_gt}, y_shift={y_shift:.3f}, closed={CLOSED_GT})")

        meta = dict(id=r["id"], terrain_seed=int(r["seed"]), scan_seed=SEED, y_shift=y_shift,
                    n_points=int(len(pts)), n_lines=int(len(z_lines)), n_pings_per_line=int(len(x_pings)),
                    swath_m=float(swath), gt_grid=n_gt, gt_closed=CLOSED_GT, sensor=sensor_params)
        with open(os.path.join(d, f"{r['id']}_scan.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[{r['split']:5s}] {r['id']:18s} points={len(pts):8d}  lines={len(z_lines)}  "
              f"swath={swath:5.1f} m  y_shift={y_shift:7.2f}  {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()