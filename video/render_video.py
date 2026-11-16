"""
Side-by-side video of a live session (online_reconstruction.py):
  [ Unity (mp4 from the Recorder) | acquired point cloud (CSV) | mesh reconstructed online ]
Without --unity-video only the last two panels.

Common clock = elapsed_s of the CSV (starts with Play, like the Recorder):
  - point cloud: at time t, all pings with elapsed_s <= t
  - mesh: at time t, the latest mesh ALREADY READY, i.e. with ready_s <= t, where
        ready_s = column 'ready_sim_s' of updates.csv if present, otherwise
        ready_s = elapsed_s + ping_interval + latency_s
    (ping_interval: wait for the next ping, which tells that the batch is complete)

Usage (from the repository root):
  python video/render_video.py --csv Recordings/scan_XXXX.csv \
      --run output/online/<date_time> --unity-video Recordings/Movie_XXXX.mp4
  Saved to output/video/video_<date_time>.mp4 (--out for another path).

Useful options: --fps 30  --panel 640 540  --zexag 2  --t-offset 0.0 (shifts the Unity video)
                --max-points-per-ping 8000  --t-end 120
"""
import argparse, glob, os, re, time
import numpy as np
import pandas as pd
import pyvista as pv
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

pv.OFF_SCREEN = True


def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(f, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def label(img, lines, f):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    y = 8
    for ln in lines:
        d.text((10, y), ln, fill=(255, 255, 255), font=f, stroke_width=2, stroke_fill=(0, 0, 0))
        y += f.size + 6
    return np.asarray(im)


def to_disp(x, y, z, zexag):
    """Unity world (x, y up, z north) -> display coordinates (x east, y north, z up)."""
    return np.c_[x, z, y * zexag]


def main():
    p = argparse.ArgumentParser(description="Video of an online reconstruction session")
    p.add_argument("--csv", required=True, help="CSV written by Unity during the session")
    p.add_argument("--run", required=True, help="online_reconstruction.py run folder (updates.csv and meshes/)")
    p.add_argument("--unity-video", default=None, help="mp4 from the Unity Recorder")
    p.add_argument("--out", default=None, help="default: output/video/video_<date_time>.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--panel", type=int, nargs=2, default=[640, 540], help="panel width and height, px")
    p.add_argument("--zexag", type=float, default=2.0, help="vertical exaggeration (display only)")
    p.add_argument("--ping-interval", type=float, default=1.0)
    p.add_argument("--t-offset", type=float, default=0.0, help="seconds added to the Unity video time")
    p.add_argument("--t-end", type=float, default=None)
    p.add_argument("--max-points-per-ping", type=int, default=8000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--view-angle", type=float, default=30.0, help="camera field of view (smaller = more zoom)")
    a = p.parse_args()
    W, H = a.panel
    f_title, f_info = font(20), font(16)

    df = pd.read_csv(a.csv)
    rng = np.random.default_rng(0)
    pings = []
    for k, g in df.groupby("ping_number", sort=True):
        if len(g) > a.max_points_per_ping:
            g = g.iloc[rng.choice(len(g), a.max_points_per_ping, replace=False)]
        pings.append((float(g.elapsed_s.max()), to_disp(g.x.to_numpy(), g.y.to_numpy(), g.z.to_numpy(), a.zexag)))
    allp = np.vstack([q for _, q in pings])
    clim = np.percentile(allp[:, 2], [2, 98])

    up = pd.read_csv(os.path.join(a.run, "updates.csv"))
    if "ready_sim_s" in up:
        up["ready_s"] = up.ready_sim_s
    else:
        up["ready_s"] = up.elapsed_s + a.ping_interval + up.latency_s
    mesh_files = {int(re.findall(r"(\d+)", os.path.basename(f))[-1]): f
                  for f in glob.glob(os.path.join(a.run, "meshes", "mesh_*.ply"))}
    up = up[up["update"].isin(list(mesh_files))].sort_values("ready_s").reset_index(drop=True)
    print(f"{len(pings)} pings, {len(up)} meshes with a snapshot")

    t_end = a.t_end or max(pings[-1][0], up.ready_s.max() if len(up) else 0) + 2.0

    reader, vfps = None, None
    if a.unity_video:
        reader = imageio.get_reader(a.unity_video)
        vfps = reader.get_meta_data()["fps"]
        vdur = reader.get_meta_data().get("duration")
        print(f"Unity video: {vfps:.1f} fps, duration {vdur} s  (data: last ping at {pings[-1][0]:.1f} s)")
        if vdur and abs(vdur - pings[-1][0]) > 2:
            print("WARNING: video duration and data time differ by > 2 s -> check Cap FPS / --t-offset")
    vframe, vidx = None, -1

    # fixed camera, the same for both 3D panels
    b = np.r_[allp.min(0), allp.max(0)]
    center = (b[:3] + b[3:]) / 2
    span = (b[3:] - b[:3]).max()
    cam = [tuple(center + np.array([0.0, -1.6 * span, 1.2 * span])), tuple(center), (0, 0, 1)]

    def plotter():
        pl = pv.Plotter(off_screen=True, window_size=(W, H))
        pl.set_background("#101820")
        pl.camera_position = cam
        pl.camera.view_angle = a.view_angle
        return pl

    pc, pm = plotter(), plotter()
    pm.add_light(pv.Light(position=tuple(center + [span, -span, 2 * span]), light_type="scene light"))
    # an invisible box pins the scene bounds, so the camera does not move
    for pl in (pc, pm):
        pl.add_mesh(pv.Box(bounds=(b[0], b[3], b[1], b[4], b[2], b[5])), opacity=0.0)
        pl.camera_position = cam
        pl.camera.view_angle = a.view_angle

    if a.out is None:
        os.makedirs("output/video", exist_ok=True)
        a.out = os.path.join("output/video", f"video_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
    else:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
    ip, im_cur, mesh_actor, n_pts = 0, -1, None, 0
    times = np.arange(0, t_end, 1 / a.fps)
    for fi, t in enumerate(times):
        new = []
        while ip < len(pings) and pings[ip][0] <= t:
            new.append(pings[ip][1]); ip += 1
        if new:
            q = np.vstack(new); n_pts += len(q)
            pc.add_points(pv.PolyData(q), scalars=q[:, 2], cmap=a.cmap, clim=clim,
                          point_size=a.point_size, render_points_as_spheres=False, show_scalar_bar=False)
        j = np.searchsorted(up.ready_s.to_numpy(), t, side="right") - 1
        if j >= 0 and j != im_cur:
            im_cur = j
            row = up.iloc[j]
            m = pv.read(mesh_files[int(row["update"])])
            v = np.asarray(m.points)
            m.points = to_disp(v[:, 0], v[:, 1], v[:, 2], a.zexag)
            if mesh_actor is not None:
                pm.remove_actor(mesh_actor)
            mesh_actor = pm.add_mesh(m, scalars=m.points[:, 2], cmap=a.cmap, clim=clim,
                                     smooth_shading=True, show_scalar_bar=False)

        img_pc = label(pc.screenshot(return_img=True),
                       ["Acquired point cloud", f"t = {t:6.1f} s   points: {n_pts:,}"], f_title)
        if im_cur >= 0:
            r = up.iloc[im_cur]
            info = f"update {int(r['update'])}  data up to {r.elapsed_s:.1f} s  latency {r.latency_s:.2f} s"
        else:
            info = "waiting for the first update"
        img_m = label(pm.screenshot(return_img=True), ["Online reconstructed mesh (Poisson)", info], f_title)
        panels = [img_pc, img_m]

        if reader is not None:
            want = int(round((t + a.t_offset) * vfps))
            try:
                while vidx < want:
                    vframe = reader.get_next_data(); vidx += 1
            except (IndexError, StopIteration):
                pass
            if vframe is None:
                vframe = np.zeros((H, W, 3), np.uint8)
            vimg = np.asarray(Image.fromarray(vframe).resize((W, H)))
            panels.insert(0, label(vimg, ["Unity: boat + VLP-16", f"t = {t:6.1f} s"], f_title))
        writer.append_data(np.hstack(panels))
        if fi % (a.fps * 10) == 0:
            print(f"  t = {t:6.1f} / {t_end:.1f} s", flush=True)
    writer.close()
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
