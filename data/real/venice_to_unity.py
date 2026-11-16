"""
Venice Lagoon bathymetry (MGDS 10.1594/IEDA/323605, ESRI ASCII 0.5 m) -> patch for a Unity Terrain + ground truth.

Output in --out:
  heightmap_<N>.raw   16-bit little-endian, N = 2^k+1 (Unity Terrain: Import Raw,
                      Depth 16 bit, Byte Order Windows)
  gt_mesh.ply         GT mesh in a local ENU frame (x=east, y=north, z=up) [m],
                      origin = SW corner of the patch, z = absolute DTM elevation
  valid_mask.npy      True where the DTM has real data (NOT filled cells): use it
                      to exclude holes from the metrics
  meta.json           size, offsets, min/max elevation -> Terrain parameters

Usage:
  python venice_to_unity.py iim_malamocco.asc --out patch_malamocco --size 256
  python venice_to_unity.py iim_malamocco.asc --out patch --size 256 --center 2310500 5025200
Without --center the patch with the most valid cells and the most relief is chosen automatically.
"""
import argparse, json
import numpy as np
import pandas as pd
from scipy import ndimage


def read_esri_ascii(path):
    header = {}
    with open(path) as f:
        for _ in range(6):
            k, v = f.readline().split()
            header[k.lower()] = float(v)
    nodata = header.get("nodata_value", -9999.0)
    z = pd.read_csv(path, skiprows=6, header=None, sep=r"\s+",
                    dtype=np.float32, engine="c").to_numpy()
    z = z[:, : int(header["ncols"])]
    valid = np.isfinite(z) & (np.abs(z - nodata) > 1e-3)
    # ESRI: row 0 = north. Flip -> row 0 = south (y grows northwards)
    z, valid = np.flipud(z), np.flipud(valid)
    x0 = header.get("xllcorner", header.get("xllcenter", 0.0))
    y0 = header.get("yllcorner", header.get("yllcenter", 0.0))
    return z, valid, x0, y0, header["cellsize"]


def auto_center(z, valid, n):
    """n x n patch with the largest valid fraction; on ties, the most relief."""
    v = valid.astype(np.float32)
    frac = ndimage.uniform_filter(v, size=n, mode="constant")
    zz = np.where(valid, z, 0.0).astype(np.float64)
    m1 = ndimage.uniform_filter(zz, n, mode="constant") / np.maximum(frac, 1e-6)
    m2 = ndimage.uniform_filter(zz**2, n, mode="constant") / np.maximum(frac, 1e-6)
    std = np.sqrt(np.maximum(m2 - m1**2, 0))
    h = n // 2
    frac[:h], frac[-h:], frac[:, :h], frac[:, -h:] = 0, 0, 0, 0
    score = np.where(frac >= frac.max() - 0.02, std, -1)
    return np.unravel_index(np.argmax(score), score.shape), float(frac.max())


def write_ply(path, z, cell):
    n_r, n_c = z.shape
    yy, xx = np.mgrid[0:n_r, 0:n_c]
    verts = np.stack([xx * cell, yy * cell, z], -1).reshape(-1, 3).astype(np.float32)
    idx = np.arange(n_r * n_c).reshape(n_r, n_c)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3),
                            np.stack([a, d, c], -1).reshape(-1, 3)])
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 f"element face {len(faces)}\nproperty list uchar int vertex_indices\n"
                 "end_header\n").encode())
        f.write(verts.tobytes())
        rec = np.empty(len(faces), dtype=[("n", "u1"), ("v", "<i4", 3)])
        rec["n"], rec["v"] = 3, faces
        f.write(rec.tobytes())


def main():
    p = argparse.ArgumentParser(description="ESRI ASCII bathymetry -> Unity heightmap + GT mesh")
    p.add_argument("asc")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=float, default=256.0, help="patch side [m]")
    p.add_argument("--center", type=float, nargs=2, help="patch centre E N in the file coordinates")
    a = p.parse_args()

    import os
    os.makedirs(a.out, exist_ok=True)
    z, valid, x0, y0, cell = read_esri_ascii(a.asc)
    print(f"grid {z.shape}, cell {cell} m, valid {valid.mean():.1%}, "
          f"z [{z[valid].min():.2f}, {z[valid].max():.2f}]")

    # N = 2^k + 1 samples -> side = (N-1)*cell
    n = int(round(a.size / cell))
    n = 1 << int(np.ceil(np.log2(max(n, 2))))
    N = n + 1
    if N > min(z.shape):
        raise SystemExit(f"patch {N}px larger than the grid {z.shape}")

    if a.center:
        cc = int(round((a.center[0] - x0) / cell))
        rc = int(round((a.center[1] - y0) / cell))
        fv = None
    else:
        (rc, cc), fv = auto_center(z, valid, N)
    r0, c0 = rc - N // 2, cc - N // 2
    r0 = int(np.clip(r0, 0, z.shape[0] - N)); c0 = int(np.clip(c0, 0, z.shape[1] - N))
    zp, vp = z[r0:r0 + N, c0:c0 + N].copy(), valid[r0:r0 + N, c0:c0 + N].copy()
    print(f"patch {N}x{N} ({n * cell:.0f} m), valid {vp.mean():.1%}"
          + (f" (best auto {fv:.1%})" if fv is not None else ""))
    if vp.mean() < 0.95:
        print("WARNING: >5% of cells without data; they are filled but excluded via valid_mask")

    # fill holes with the nearest valid value (only to get a closed surface)
    if not vp.all():
        _, (ir, ic) = ndimage.distance_transform_edt(~vp, return_indices=True)
        zp = zp[ir, ic]

    zmin, zmax = float(zp.min()), float(zp.max())
    span = max(zmax - zmin, 1e-3)
    h16 = np.round((zp - zmin) / span * 65535).astype("<u2")
    # Unity: RAW row 0 = z=0 (south). If the Terrain comes out mirrored, use
    # "Flip Vertically" on import and check with an asymmetric feature.
    h16.tofile(os.path.join(a.out, f"heightmap_{N}.raw"))
    np.save(os.path.join(a.out, "valid_mask.npy"), vp)
    write_ply(os.path.join(a.out, "gt_mesh.ply"), zp, cell)

    meta = dict(source=os.path.basename(a.asc), cell_m=cell, samples=N,
                side_m=n * cell, origin_E=x0 + c0 * cell, origin_N=y0 + r0 * cell,
                z_min=zmin, z_max=zmax, valid_fraction=float(vp.mean()),
                unity=dict(heightmap_resolution=N, terrain_width=n * cell,
                           terrain_length=n * cell, terrain_height=span,
                           terrain_position_y=zmin,
                           note="ENU (x,y,z) -> Unity (x, z_up, y_north)"))
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta["unity"], indent=2))


if __name__ == "__main__":
    main()
