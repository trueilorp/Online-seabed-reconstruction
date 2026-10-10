"""
Generatore di fondali sintetici (heightmap) per dataset train/val/test.

Per ogni terreno produce:
  <out>/<split>/<id>/<id>.raw          heightmap uint16 little-endian ("Windows" in Unity), da importare nel Terrain
  <out>/<split>/<id>/<id>.png          stessa heightmap in PNG 16 bit (se Pillow e' installato)
  <out>/<split>/<id>/<id>_preview.png  hillshade per un controllo visivo (se matplotlib e' installato)
  <out>/<split>/<id>/<id>.json         seed, famiglia, parametri, oggetti aggiunti, impostazioni Unity
  <out>/manifest.csv                   una riga per terreno, con lo split assegnato

Riproducibilita': ogni terreno dipende solo da (famiglia, seed, risoluzione, size, height_range, flag).
Lo split e' fatto PER TERRENO (mai per scan): un terreno sta in un solo split.

Nota sul rumore: e' sintetizzato nello spazio di Fourier, quindi e' periodico (il bordo sinistro si
raccorda con il destro). Non e' un problema per il Terrain, ma tienilo presente se vuoi tassellare.

Uso minimo:
  python generate_seabed_heightmaps.py --out data/synthetic --n-per-family 10
"""
import argparse
import csv
import json
import os
from dataclasses import dataclass

import numpy as np

try:
    from PIL import Image
except ImportError:  # PNG a 16 bit opzionale
    Image = None

try:
    import matplotlib
    matplotlib.use("Agg")  # headless (WSL senza display)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    res: int
    size_m: float
    rng: np.random.Generator
    X: np.ndarray  # coordinate x in metri (varia lungo le colonne)
    Y: np.ndarray  # coordinate y in metri (varia lungo le righe)

    @property
    def cell(self) -> float:
        return self.size_m / (self.res - 1)

    def noise(self, beta: float, min_wavelength_m: float) -> np.ndarray:
        """Rumore spettrale privo di strutture piu' fini di ~min_wavelength_m."""
        return spectral_noise(self.res, beta, self.rng, fc=self.cell / min_wavelength_m)


def spectral_noise(res: int, beta: float, rng: np.random.Generator, fc: float = None) -> np.ndarray:
    """Rumore gaussiano con spettro di potenza ~ 1/f^beta, normalizzato a std = 1.
    beta piu' alto = piu' liscio. fc = frequenza di taglio gaussiana in cicli/pixel (vedi Ctx.fc):
    senza taglio la pendenza locale e' dominata dal rumore a scala del pixel (irrealistico)."""
    white = rng.standard_normal((res, res))
    fx = np.fft.fftfreq(res)[:, None]
    fy = np.fft.fftfreq(res)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    filt = f ** (-beta / 2.0)
    if fc is not None:
        filt = filt * np.exp(-(f / fc) ** 2)
    filt[0, 0] = 0.0
    out = np.fft.ifft2(np.fft.fft2(white) * filt).real
    return out / out.std()


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _idx(ctx: Ctx, cx: float, cy: float):
    i = int(np.clip(round(cy / ctx.cell), 0, ctx.res - 1))  # riga <-> y
    j = int(np.clip(round(cx / ctx.cell), 0, ctx.res - 1))  # colonna <-> x
    return i, j


def add_boulders(h: np.ndarray, ctx: Ctx, n: int, r_range=(0.3, 1.2)) -> int:
    """Massi: semisfere schiacciate, appoggiate sul terreno locale (in place)."""
    rng = ctx.rng
    for _ in range(n):
        cx, cy = rng.uniform(0, ctx.size_m, 2)
        r = rng.uniform(*r_range)
        i, j = _idx(ctx, cx, cy)
        rc = int(np.ceil(r / ctx.cell)) + 1
        i0, i1 = max(0, i - rc), min(ctx.res, i + rc + 1)
        j0, j1 = max(0, j - rc), min(ctx.res, j + rc + 1)
        d2 = (ctx.X[i0:i1, j0:j1] - cx) ** 2 + (ctx.Y[i0:i1, j0:j1] - cy) ** 2
        bump = 0.8 * np.sqrt(np.maximum(r ** 2 - d2, 0.0))
        base = h[i, j] - 0.3 * r
        mask = d2 < r ** 2
        win = h[i0:i1, j0:j1]
        win[mask] = np.maximum(win[mask], base + bump[mask])
    return n


# ---------------------------------------------------------------------------
# Famiglie di fondale (tutte restituiscono altezze in metri + parametri usati)
# ---------------------------------------------------------------------------

def fam_flat_sand(ctx: Ctx):
    rng = ctx.rng
    amp = rng.uniform(0.03, 0.12)
    tilt_deg = rng.uniform(0.0, 1.0)
    az = rng.uniform(0, 2 * np.pi)
    h = amp * ctx.noise(2.6, 4.0)
    h += np.tan(np.deg2rad(tilt_deg)) * (ctx.X * np.cos(az) + ctx.Y * np.sin(az))
    return h, dict(amp_m=amp, tilt_deg=tilt_deg)


def fam_sand_ripples(ctx: Ctx):
    rng = ctx.rng
    wavelength = rng.uniform(1.0, 3.0)
    amp = wavelength * rng.uniform(0.01, 0.04)  # steepness realistica (pendenza picco ~4-14 gradi)
    theta = rng.uniform(0, np.pi)
    warp = 0.6 * wavelength * ctx.noise(3.0, 8.0)
    u = ctx.X * np.cos(theta) + ctx.Y * np.sin(theta) + warp
    env = smoothstep(0.5 + 0.35 * ctx.noise(3.0, 8.0))  # ampiezza ondulazioni variabile
    ripples = amp * env * np.sin(2 * np.pi * u / wavelength)
    swell = 0.4 * ctx.noise(3.0, 8.0)
    return ripples + swell, dict(wavelength_m=wavelength, amp_m=amp, theta_rad=theta)


def fam_rocky(ctx: Ctx):
    rng = ctx.rng
    n = ctx.noise(2.6, 2.0)
    ridged = np.clip(1.0 - np.abs(n) / 2.5, 0.0, 1.0) ** 2
    amp = rng.uniform(1.0, 3.0)
    h = amp * ridged + 0.06 * ctx.noise(2.4, 0.8)
    n_boulders = add_boulders(h, ctx, int(rng.integers(20, 60)))
    return h, dict(amp_m=amp, n_boulders=n_boulders)


def fam_slope(ctx: Ctx):
    rng = ctx.rng
    slope_deg = rng.uniform(1.0, 8.0)
    az = rng.uniform(0, 2 * np.pi)
    noise_amp = rng.uniform(0.2, 0.6)
    h = np.tan(np.deg2rad(slope_deg)) * (ctx.X * np.cos(az) + ctx.Y * np.sin(az))
    h += noise_amp * ctx.noise(2.6, 3.0)
    return h, dict(slope_deg=slope_deg, azimuth_rad=az, noise_amp_m=noise_amp)


def fam_ridges(ctx: Ctx):
    rng = ctx.rng
    amp = rng.uniform(2.0, 5.0)
    n = ctx.noise(2.8, 6.0)
    ridged = np.clip(1.0 - np.abs(n) / 2.5, 0.0, 1.0) ** 2
    h = amp * ridged + 0.1 * ctx.noise(2.4, 1.5)
    return h, dict(amp_m=amp)


def fam_mixed(ctx: Ctx):
    """Sabbia rippled e roccia fuse con una maschera liscia."""
    h_sand, p_sand = fam_sand_ripples(ctx)
    h_rock, p_rock = fam_rocky(ctx)
    mask = smoothstep(0.5 + 0.8 * ctx.noise(3.0, 10.0))
    h = mask * h_rock + (1.0 - mask) * h_sand
    return h, dict(sand=p_sand, rock=p_rock, rock_fraction=float(mask.mean()))


FAMILIES = {
    "flat_sand": fam_flat_sand,
    "sand_ripples": fam_sand_ripples,
    "rocky": fam_rocky,
    "slope": fam_slope,
    "ridges": fam_ridges,
    "mixed": fam_mixed,
}


# ---------------------------------------------------------------------------
# Strutture (obiettivo finale: tubi/cavi, piloni) -- opzionali
# ---------------------------------------------------------------------------

def _dist_point_segment(X, Y, p0, p1):
    d = p1 - p0
    t = ((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / (d @ d)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(X - (p0[0] + t * d[0]), Y - (p0[1] + t * d[1]))


def add_structures(h: np.ndarray, ctx: Ctx):
    """Aggiunge 0-2 tubi (semicilindri appoggiati) e 0-2 piloni (cilindri). Dopo il fit del terreno."""
    rng = ctx.rng
    info = dict(pipes=[], pillars=[])
    for _ in range(int(rng.integers(0, 3))):
        p0 = rng.uniform(0, ctx.size_m, 2)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.4, 0.9) * ctx.size_m
        p1 = p0 + length * np.array([np.cos(ang), np.sin(ang)])
        r = rng.uniform(0.15, 0.5)
        d = _dist_point_segment(ctx.X, ctx.Y, p0, p1)
        h += np.sqrt(np.maximum(r ** 2 - d ** 2, 0.0))  # segue il terreno locale
        info["pipes"].append(dict(p0=p0.tolist(), p1=p1.tolist(), radius_m=r))
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = rng.uniform(0.1, 0.9, 2) * ctx.size_m
        r = rng.uniform(0.4, 1.0)
        height = rng.uniform(2.0, 4.0)
        i, j = _idx(ctx, cx, cy)
        base = h[i, j]
        mask = np.hypot(ctx.X - cx, ctx.Y - cy) < r
        h[mask] = np.maximum(h[mask], base + height)
        info["pillars"].append(dict(center=[cx, cy], radius_m=r, height_m=height))
    return h, info


# ---------------------------------------------------------------------------
# Terreno completo, I/O, split
# ---------------------------------------------------------------------------

def fit_to_range(h: np.ndarray, height_range: float, lo=0.02, hi=0.85):
    """Sposta il minimo a lo*H; se serve, comprime uniformemente per stare sotto hi*H
    (lascia margine per le strutture). Ritorna anche il fattore di scala applicato."""
    floor = lo * height_range
    h = h - h.min() + floor
    scale = 1.0
    if h.max() > hi * height_range:
        scale = (hi * height_range - floor) / (h.max() - floor)
        h = floor + (h - floor) * scale
    return h, float(scale)


def generate_terrain(family: str, seed: int, res: int, size_m: float, height_range: float,
                     with_structures: bool):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, size_m, res)
    X, Y = np.meshgrid(xs, xs)
    ctx = Ctx(res=res, size_m=size_m, rng=rng, X=X, Y=Y)

    h, params = FAMILIES[family](ctx)
    h, scale = fit_to_range(h, height_range)
    structures = {}
    if with_structures:
        h, structures = add_structures(h, ctx)
    clipped = bool(h.max() > height_range)
    meta = dict(family=family, seed=seed, resolution=res, size_m=size_m, height_range_m=height_range,
                scale_applied=scale, clipped=clipped, params=params, structures=structures,
                relief_m=float(h.max() - h.min()))
    return h, meta


def to_uint16(h: np.ndarray, height_range: float) -> np.ndarray:
    return np.clip(np.round(h / height_range * 65535.0), 0, 65535).astype("<u2")


def save_preview(h: np.ndarray, cell: float, path: str):
    gy, gx = np.gradient(h, cell)
    light = np.array([-1.0, 1.0, 1.2])
    light /= np.linalg.norm(light)
    shade = (-gx * light[0] - gy * light[1] + light[2]) / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    plt.imsave(path, np.clip(shade, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0)


def assign_splits(n: int, ratios, test_only: bool):
    if test_only:
        return ["test"] * n
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    n_te = n - n_tr - n_va
    if n >= 3:  # almeno un terreno per split
        n_va, n_te = max(n_va, 1), max(n_te, 1)
        n_tr = n - n_va - n_te
    return ["train"] * n_tr + ["val"] * n_va + ["test"] * max(n_te, 0)


def main():
    ap = argparse.ArgumentParser(description="Genera fondali sintetici (heightmap RAW 16 bit) con split per terreno.")
    ap.add_argument("--out", default="data/synthetic", help="cartella di output")
    ap.add_argument("--n-per-family", type=int, default=10, help="terreni per ogni famiglia")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--test-only-families", nargs="*", default=[], choices=list(FAMILIES),
                    help="famiglie usate SOLO nel test (per misurare la generalizzazione a fondali mai visti)")
    ap.add_argument("--resolution", type=int, default=1025, help="lato della heightmap; per Unity usa 2^n + 1 (513, 1025, ...)")
    ap.add_argument("--size", type=float, default=100.0, help="lato del terreno in metri (X e Z)")
    ap.add_argument("--height-range", type=float, default=20.0, help="escursione verticale massima in metri (= Terrain Height in Unity)")
    ap.add_argument("--seed", type=int, default=0, help="seed globale del dataset")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--with-structures", action="store_true", help="aggiunge tubi e piloni")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    if abs(sum(args.split) - 1.0) > 1e-6:
        ap.error("le tre frazioni di --split devono sommare a 1")

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    cell = args.size / (args.resolution - 1)

    for fam_idx, family in enumerate(args.families):
        splits = assign_splits(args.n_per_family, args.split, family in args.test_only_families)
        for k, split in enumerate(splits):
            seed = int(np.random.SeedSequence([args.seed, fam_idx, k]).generate_state(1)[0])
            tid = f"{family}_{k:03d}"
            h, meta = generate_terrain(family, seed, args.resolution, args.size, args.height_range,
                                       args.with_structures)
            meta.update(id=tid, split=split,
                        unity=dict(import_raw=dict(depth="16 bit", byte_order="Windows",
                                                   width=args.resolution, height=args.resolution),
                                   terrain_size=dict(x=args.size, y=args.height_range, z=args.size)))

            d = os.path.join(args.out, split, tid)
            os.makedirs(d, exist_ok=True)
            arr = to_uint16(h, args.height_range)
            arr.tofile(os.path.join(d, f"{tid}.raw"))
            if Image is not None:
                Image.fromarray(arr.astype(np.uint16)).save(os.path.join(d, f"{tid}.png"))
            if plt is not None and not args.no_preview:
                save_preview(h, cell, os.path.join(d, f"{tid}_preview.png"))
            with open(os.path.join(d, f"{tid}.json"), "w") as f:
                json.dump(meta, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

            manifest.append(dict(id=tid, family=family, split=split, seed=seed,
                                 resolution=args.resolution, size_m=args.size,
                                 height_range_m=args.height_range, relief_m=round(meta["relief_m"], 3),
                                 scale_applied=round(meta["scale_applied"], 4), clipped=meta["clipped"],
                                 n_pipes=len(meta["structures"].get("pipes", [])),
                                 n_pillars=len(meta["structures"].get("pillars", [])),
                                 raw=os.path.join(split, tid, f"{tid}.raw")))
            print(f"[{split:5s}] {tid}  relief={meta['relief_m']:.2f} m  scala={meta['scale_applied']:.3f}"
                  f"{'  CLIPPED' if meta['clipped'] else ''}")

    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    n = {s: sum(m["split"] == s for m in manifest) for s in ("train", "val", "test")}
    print(f"\nGenerati {len(manifest)} terreni -> {args.out}  (train {n['train']}, val {n['val']}, test {n['test']})")
    print(f"Unity: Terrain Settings > Import Raw: Depth 16 bit, Byte Order Windows, "
          f"Resolution {args.resolution} x {args.resolution}; "
          f"Terrain Width/Length {args.size} m, Height {args.height_range} m.")
    n_scaled = sum(m["scale_applied"] < 1.0 for m in manifest)
    if n_scaled:
        print(f"Nota: {n_scaled} terreni sono stati compressi in verticale per stare nel range "
              f"(vedi scale_applied nel manifest).")


if __name__ == "__main__":
    main()