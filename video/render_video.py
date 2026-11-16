"""
Result video of an online reconstruction session (online_reconstruction.py), 1920x1080 H.264:
  top:    [ Unity recording | acquired point cloud | online Poisson mesh ]   (same camera for the two 3D panels)
  bottom: [ plan view: GT hillshade, reconstructed area, boat track, last rebuilt box | metrics and latency ]
with a title card at the start and a summary at the end. Common clock: elapsed_s of the CSV (simulation time).

Everything shown is read from saved files: the Unity CSV, <run>/updates.csv, <run>/meshes/, <run>/config.txt,
<run>/metrics_online.csv (evaluate_online.py --tau 0.2 0.5) and the GT patch. The per-update table behind the
plots and the summary is written to <run>/metrics_per_update.csv.

Mesh delay = time the mesh is ready - time of its newest data, on the simulation clock. With the wall-clock
timestamps of updates.csv (arrived_unix_s, ready_unix_s) the wall clock is mapped to the simulation clock by a
linear fit on the batch arrivals that did not wait in a queue (Unity can run slower than real time); without them
ready = max(data time + 1 ping, previous ready) + latency (sequential processing).

Boat track: the CSV has no boat position; each ping is a swath symmetric about the boat, so the centre of its
footprint is used as the nadir.

Usage (from the repository root):
	python video/render_video.py --run output/online/<run> --unity-video <Recordings/Movie_XXX.mp4> --preview
	python video/render_video.py --run output/online/<run> --unity-video <Recordings/Movie_XXX.mp4>
Output: output/video/video_<date_time>.mp4 (--out for another path); --preview writes PNG frames instead.
"""
import argparse
import functools
import glob
import json
import os
import re
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import yaml
from matplotlib import colormaps, font_manager
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw, ImageFont

pv.OFF_SCREEN = True

W, H = 1920, 1080
TITLE_H, FOOT_H, TOP_H, PANEL_W = 40, 30, 560, 640
BOT_Y = TITLE_H + TOP_H
BOT_H = H - BOT_Y - FOOT_H                     # 450
BG, FG, MUTED, GRID = "#0f1419", "#e6e6e6", "#8b949e", "#2a313a"
ACCENT = "#ffffff"                             # current ping, boat position, time cursor
MESH_COLOR = (0.62, 0.70, 0.78)                # as visualization/screenshot_run.py
OLD_POINTS, AREA = "#6b7280", "#4fa3d9"
SERIES = ["#6ea8fe", "#c3a6ff"]                # F-score at the two tau
CHAMFER, LATENCY = "#7dd3c0", "#6ea8fe"
CMAP = "viridis_r"                             # depth: shallow yellow, deep purple
TITLE = "Online seabed reconstruction - Malamocco inlet (Venice Lagoon), simulated survey"
VIEW_ANGLE, ELEVATION, AZIMUTH, FILL, FILL_Y = 30.0, 35.0, 0.0, 0.88, 0.62


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False):
	path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
	return ImageFont.truetype(path, size)


def to_disp(x, y, z, zexag):
	"""Unity world (x, y up, z north) -> display coordinates (x east, y north, z up, exaggerated)."""
	return np.c_[x, z, y * zexag]


# ---------------------------------------------------------------- data

def load_pings(csv_path: str, zexag: float, max_points: int):
	"""Per ping: time, number of points, footprint box and centre (nadir), display subsample and its depth."""
	df = pd.read_csv(csv_path, usecols=["x", "y", "z", "ping_number", "elapsed_s"])
	rng = np.random.default_rng(0)
	pings = []
	for k, g in df.groupby("ping_number", sort=True):
		x, y, z = g["x"].to_numpy(), g["y"].to_numpy(), g["z"].to_numpy()
		idx = rng.choice(len(g), max_points, replace=False) if len(g) > max_points else slice(None)
		pings.append(dict(ping=int(k), t=float(g["elapsed_s"].max()), n=len(g),
						  box=(x.min(), z.min(), x.max(), z.max()),
						  nadir=((x.min() + x.max()) / 2, (z.min() + z.max()) / 2),
						  xyz=to_disp(x[idx], y[idx], z[idx], zexag), depth=-y[idx]))
	return pings


def ready_sim_times(up: pd.DataFrame, ping_dt: float, full: pd.Series):
	"""Simulation time at which each mesh was ready, and the simulation/wall clock rate used."""
	if {"arrived_unix_s", "ready_unix_s"} <= set(up.columns):
		prev_ready = up["ready_unix_s"].shift()
		waited = prev_ready.isna() | (up["arrived_unix_s"] > prev_ready + 0.05)
		fit = up[waited & full]
		if len(fit) >= 2:
			rate, offset = np.polyfit(fit["arrived_unix_s"], fit["elapsed_s"] + ping_dt, 1)
			ready = offset + rate * up["ready_unix_s"].to_numpy()
			return np.maximum(ready, up["elapsed_s"].to_numpy() + ping_dt), float(rate)
	ready, prev = [], -np.inf
	for elapsed, latency in zip(up["elapsed_s"], up["latency_s"]):
		prev = max(elapsed + ping_dt, prev) + latency
		ready.append(prev)
	return np.array(ready), 1.0


def update_table(run: str, pings: list, taus: list, every: int):
	"""updates.csv + simulation-clock ready time, mesh delay, box, metrics at each tau (metrics_online.csv)."""
	up = pd.read_csv(os.path.join(run, "updates.csv"))
	ping_dt = float(np.median(np.diff([p["t"] for p in pings]))) if len(pings) > 1 else 1.0
	full = (up["last_ping"] - up["first_ping"] + 1) >= every
	up["ready_sim_s"], rate = ready_sim_times(up, ping_dt, full)
	up["mesh_delay_s"] = up["ready_sim_s"] - up["elapsed_s"]
	up["full_batch"] = full

	by_ping = {p["ping"]: p["box"] for p in pings}
	boxes = [np.array([by_ping[i] for i in range(a, b + 1) if i in by_ping]) for a, b in zip(up["first_ping"], up["last_ping"])]
	up["box_x0"], up["box_z0"] = [b[:, 0].min() for b in boxes], [b[:, 1].min() for b in boxes]
	up["box_x1"], up["box_z1"] = [b[:, 2].max() for b in boxes], [b[:, 3].max() for b in boxes]

	path = os.path.join(run, "metrics_online.csv")
	hint = f"python evaluate_online.py {run} --tau {' '.join(f'{t:g}' for t in taus)}"
	if not os.path.exists(path):
		raise SystemExit(f"No metrics_online.csv in {run}: run first\n  {hint}")
	mo = pd.read_csv(path)
	for tau in taus:
		sel = mo[np.isclose(mo["tau"], tau)].set_index("update")
		if sel.empty:
			raise SystemExit(f"metrics_online.csv has no tau = {tau:g}: run\n  {hint}")
		for m in ("precision", "recall", "fscore"):
			up[f"{m}_{tau:g}"] = up["update"].map(sel[m])
		up["chamfer"] = up["update"].map(sel["chamfer"])
	return up, rate, ping_dt


# ---------------------------------------------------------------- 3D panels

def project(points, cam, focal, aspect):
	"""Pinhole projection as VTK (vertical view angle), z up -> normalized device coordinates."""
	f = (focal - cam) / np.linalg.norm(focal - cam)
	r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
	u = np.cross(r, f)
	v = np.atleast_2d(points) - cam
	t = np.tan(np.radians(VIEW_ANGLE) / 2)
	return np.c_[(v @ r) / ((v @ f) * t * aspect), (v @ u) / ((v @ f) * t)]


def fit_camera(points, aspect):
	"""Oblique camera (ELEVATION, AZIMUTH) at the distance that makes the points fill FILL of the panel width
	and FILL_Y of its height (room for the text at the top and bottom)."""
	center = (points.min(axis=0) + points.max(axis=0)) / 2
	el, az = np.radians(ELEVATION), np.radians(AZIMUTH)
	d = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])
	span = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
	a, b = 0.2 * span, 50 * span
	for _ in range(60):
		mid = (a + b) / 2
		ndc = project(points, center + mid * d, center, aspect)
		if max(np.abs(ndc[:, 0]).max() / FILL, np.abs(ndc[:, 1]).max() / FILL_Y) > 1:
			a = mid
		else:
			b = mid
	return center + b * d, center


class Scene:
	"""Off-screen 3D panel with a fixed camera, a scale bar and a north arrow."""

	def __init__(self, content):
		"""content: display points of the final reconstruction (the camera is fitted on them)."""
		self.pl = pv.Plotter(off_screen=True, window_size=(PANEL_W, TOP_H))
		self.pl.set_background(BG)
		lo, hi = content.min(axis=0), content.max(axis=0)
		span = hi - lo
		self.bar_len = next(L for L in (100, 50, 20, 10, 5) if L <= 0.45 * span[0])
		y_bar, z0 = lo[1] - 0.04 * span[1], float(np.median(content[:, 2]))  # at the typical seabed height
		self.bar = np.array([[lo[0], y_bar, z0], [lo[0] + self.bar_len, y_bar, z0]])
		arrow_len = 0.12 * max(span[0], span[1])
		self.arrow = np.array([[hi[0] - 0.02 * span[0], y_bar, z0], [hi[0] - 0.02 * span[0], y_bar + arrow_len, z0]])
		fit_lo, fit_hi = lo.copy(), hi.copy()
		fit_lo[1] = y_bar - 0.02 * span[1]
		self.cam, self.focal = fit_camera(np.vstack([content, self.bar, self.arrow]), PANEL_W / TOP_H)

		self.pl.add_mesh(pv.Box(bounds=(fit_lo[0], fit_hi[0], fit_lo[1], fit_hi[1], fit_lo[2], fit_hi[2])), opacity=0.0)
		tick = 0.015 * span[0]
		for a, b in [(self.bar[0], self.bar[1]), (self.bar[0], self.bar[0] + [0, 0, tick]),
					 (self.bar[1], self.bar[1] + [0, 0, tick])]:
			self.pl.add_mesh(pv.Line(a, b), color=FG, line_width=3)
		self.pl.add_mesh(pv.Arrow(start=self.arrow[0], direction=(0, 1, 0), scale=arrow_len, tip_length=0.3,
								  tip_radius=0.12, shaft_radius=0.035), color=FG)
		self.pl.camera_position = [tuple(self.cam), tuple(self.focal), (0, 0, 1)]
		self.pl.camera.view_angle = VIEW_ANGLE

	def px(self, points):
		ndc = project(points, self.cam, self.focal, PANEL_W / TOP_H)
		return np.c_[(ndc[:, 0] + 1) / 2 * PANEL_W, (1 - ndc[:, 1]) / 2 * TOP_H]

	def render(self) -> Image.Image:
		self.pl.renderer.ResetCameraClippingRange()
		img = Image.fromarray(self.pl.screenshot(return_img=True))
		d = ImageDraw.Draw(img)
		mid = self.px(self.bar.mean(axis=0))[0]
		d.text((mid[0], mid[1] + 14), f"{self.bar_len} m", fill=FG, font=font(13), anchor="mt")
		tip = self.px(self.arrow[1])[0]
		d.text((tip[0], tip[1] - 8), "N", fill=FG, font=font(14, True), anchor="mb")
		return img


class CloudScene(Scene):
	"""Coloured = last `every` pings (current update window), grey = earlier pings (still in the voxel map),
	accent = newest ping for 1 s."""

	def __init__(self, content, pings, every, clim, point_size):
		super().__init__(content)
		self.pings, self.every, self.clim, self.ps = pings, every, clim, point_size
		self.times = np.array([p["t"] for p in pings])
		self.key, self.n_old = None, -1

	def set_time(self, t):
		i1 = int(np.searchsorted(self.times, t, side="right")) - 1
		highlight = i1 >= 0 and t - self.times[i1] < 1.0
		if (i1, highlight) == self.key:
			return
		if self.key is None or i1 != self.key[0]:
			i0 = max(0, i1 - self.every + 1)
			n_old = i0 if i1 >= 0 else 0
			if n_old != self.n_old:
				self.pl.remove_actor("old")
				if n_old:
					old = np.vstack([p["xyz"] for p in self.pings[:n_old]])
					self.pl.add_points(pv.PolyData(old), color=OLD_POINTS, opacity=0.25, point_size=self.ps * 0.75,
									   name="old", render_points_as_spheres=False)
				self.n_old = n_old
			self.pl.remove_actor("window")
			if i1 >= 0:
				win = self.pings[i0:i1 + 1]
				self.pl.add_points(pv.PolyData(np.vstack([p["xyz"] for p in win])),
								   scalars=np.concatenate([p["depth"] for p in win]), cmap=CMAP, clim=self.clim,
								   point_size=self.ps, show_scalar_bar=False, name="window",
								   render_points_as_spheres=False)
		self.pl.remove_actor("ping")
		if highlight:
			self.pl.add_points(pv.PolyData(self.pings[i1]["xyz"]), color=ACCENT, point_size=self.ps + 1, name="ping",
							   render_points_as_spheres=False)
		self.key = (i1, highlight)


class MeshScene(Scene):
	def __init__(self, content, snapshots, zexag):
		super().__init__(content)
		self.snapshots, self.zexag, self.k = snapshots, zexag, None

	def set_update(self, k):
		if k == self.k:
			return
		self.pl.remove_actor("mesh")
		if k is not None:
			m = pv.read(self.snapshots[k])
			v = np.asarray(m.points)
			m.points = to_disp(v[:, 0], v[:, 1], v[:, 2], self.zexag)
			self.pl.add_mesh(m, color=MESH_COLOR, smooth_shading=True, ambient=0.25, diffuse=0.75, specular=0.1,
							 name="mesh")
		self.k = k


# ---------------------------------------------------------------- bottom panels

def style_axes(ax):
	ax.set_facecolor(BG)
	for s in ax.spines.values():
		s.set_color(GRID)
	ax.tick_params(colors=MUTED, labelsize=9)
	ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)


def figure_image(fig) -> Image.Image:
	fig.canvas.draw()
	return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


class PlanView:
	"""Plan view: GT hillshade, area covered by the shown mesh, last rebuilt box (one cached image per update);
	the boat track and position are drawn per frame."""

	def __init__(self, gt_dir, terrain_xz, up, snapshots, pings):
		meta = json.load(open(os.path.join(gt_dir, "meta.json")))
		cell, n = meta["cell_m"], meta["samples"]
		gt = pv.read(os.path.join(gt_dir, "gt_mesh.ply"))
		elevation = np.asarray(gt.points)[:, 2].reshape(n, n)            # row = north, column = east
		valid = np.load(os.path.join(gt_dir, "valid_mask.npy"))
		shade = LightSource(azdeg=315, altdeg=45).hillshade(elevation, vert_exag=3, dx=cell, dy=cell)
		shade = np.where(valid, shade, np.nan)
		px, pz = terrain_xz
		self.extent = [px - cell / 2, px + (n - 1) * cell + cell / 2, pz - cell / 2, pz + (n - 1) * cell + cell / 2]
		self.cell, self.n, self.origin = cell, n, (px - cell / 2, pz - cell / 2)

		self.masks = {}
		for k, path in snapshots.items():
			c = np.asarray(pv.read(path).cell_centers().points)
			cols = np.floor((c[:, 0] - self.origin[0]) / cell).astype(int)
			rows = np.floor((c[:, 2] - self.origin[1]) / cell).astype(int)
			ok = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
			mask = np.zeros((n, n), dtype=bool)
			mask[rows[ok], cols[ok]] = True
			self.masks[k] = mask
		final = self.masks[max(self.masks)]
		rr, cc = np.nonzero(final)
		margin = 15
		x0, x1 = self.origin[0] + cc.min() * cell - margin, self.origin[0] + (cc.max() + 1) * cell + margin
		z0, z1 = self.origin[1] + rr.min() * cell - margin, self.origin[1] + (rr.max() + 1) * cell + margin

		self.fig = plt.figure(figsize=(W / 2 / 100, BOT_H / 100), dpi=100, facecolor=BG)
		self.ax = self.fig.add_axes([0.09, 0.11, 0.5, 0.80])
		style_axes(self.ax)
		self.ax.imshow(shade, origin="lower", extent=self.extent, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
		self.ax.set_xlim(x0, x1); self.ax.set_ylim(z0, z1)
		self.ax.set_aspect("equal", adjustable="box")
		self.ax.set_anchor("W")
		self.ax.set_xlabel("x [m] (east)", fontsize=10); self.ax.set_ylabel("z [m] (north)", fontsize=10)
		self.fig.text(0.08, 0.95, "Plan view", color=FG, fontsize=12, fontweight="bold", va="top")
		handles = [Patch(facecolor="#9a9a9a", label=f"GT hillshade ({cell:g} m DTM)"),
				   Patch(facecolor=AREA, alpha=0.45, edgecolor=AREA, label="area covered by the mesh"),
				   Line2D([], [], color=FG, lw=1.2, label="boat track (footprint centre)"),
				   Line2D([], [], color=ACCENT, marker="o", lw=0, markeredgecolor=BG, label="boat position"),
				   Patch(facecolor="none", edgecolor=FG, linestyle="--", label="box rebuilt at the last update")]
		self.fig.canvas.draw()
		right = self.ax.get_position().x1
		self.fig.legend(handles=handles, loc="center left", bbox_to_anchor=(right + 0.03, 0.5), frameon=False,
						labelcolor=FG, fontsize=10)
		self.overlays, self.cache = [], {}
		self.up = up.set_index("update")
		self.fig.canvas.draw()
		self.times = np.array([p["t"] for p in pings])
		self.nadirs = np.array([p["nadir"] for p in pings])

	def to_px(self, xz):
		p = self.ax.transData.transform(np.atleast_2d(xz))
		return np.c_[p[:, 0], BOT_H - p[:, 1]]

	def base(self, k):
		if k not in self.cache:
			for a in self.overlays:
				a.remove()
			self.overlays = []
			if k is not None:
				rgba = np.zeros((self.n, self.n, 4))
				rgba[self.masks[k]] = matplotlib.colors.to_rgba(AREA, 0.45)
				self.overlays.append(self.ax.imshow(rgba, origin="lower", extent=self.extent, interpolation="nearest"))
				self.overlays.append(self.ax.contour(self.masks[k].astype(float), levels=[0.5], colors=[AREA],
													 linewidths=1.0, origin="lower", extent=self.extent))
				b = self.up.loc[k]
				self.overlays.append(self.ax.add_patch(Rectangle((b["box_x0"], b["box_z0"]), b["box_x1"] - b["box_x0"],
																 b["box_z1"] - b["box_z0"], fill=False, ec=FG,
																 ls="--", lw=1.2)))
			self.cache[k] = figure_image(self.fig)
		return self.cache[k]

	def render(self, t, k) -> Image.Image:
		img = self.base(k).copy()
		i1 = int(np.searchsorted(self.times, t, side="right")) - 1
		if i1 >= 0:
			pos = np.array([np.interp(t, self.times, self.nadirs[:, j]) for j in (0, 1)])
			track = self.to_px(np.vstack([self.nadirs[:i1 + 1], pos]))
			d = ImageDraw.Draw(img)
			d.line([tuple(p) for p in track], fill=FG, width=2)
			x, y = self.to_px(pos)[0]
			d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=ACCENT, outline=BG, width=2)
		return img


class Charts:
	"""F-score at each tau, Chamfer and latency per update (at the time each mesh was ready); one cached image
	per update, the time cursor is drawn per frame."""

	def __init__(self, shown, taus, t_end, every, ping_dt):
		self.shown, self.taus = shown.reset_index(drop=True), taus
		self.fig = plt.figure(figsize=(W / 2 / 100, BOT_H / 100), dpi=100, facecolor=BG)
		left, width = 0.09, 0.72
		self.axes = [self.fig.add_axes([left, y, width, 0.22]) for y in (0.68, 0.40, 0.12)]
		f_min = min(self.shown[f"fscore_{t:g}"].min() for t in taus)
		budget = float(self.shown.loc[self.shown["full_batch"], "budget_s"].median())
		limits = [(max(0.0, np.floor(f_min * 20) / 20 - 0.05), 1.0), (0, self.shown["chamfer"].max() * 1.25),
				  (0, max(self.shown["latency_s"].max(), budget) * 1.3)]
		labels = ["F-score", "Chamfer [m]", "latency [s]"]
		for ax, lim, lab in zip(self.axes, limits, labels):
			style_axes(ax)
			ax.set_xlim(0, t_end); ax.set_ylim(*lim)
			ax.set_ylabel(lab, fontsize=10)
			ax.grid(axis="y", color=GRID, lw=0.6)
		for ax in self.axes[:2]:
			ax.tick_params(labelbottom=False)
		self.axes[2].set_xlabel("simulation time [s]", fontsize=10)
		self.axes[2].axhline(budget, color=MUTED, ls="--", lw=1)
		self.axes[2].text(0.005 * t_end, budget, f"budget {budget:.1f} s ({every} pings)", color=MUTED, fontsize=9,
						  va="bottom", ha="left")
		self.fig.text(left, 0.97, "Quality vs GT and timing per update", color=FG, fontsize=12, fontweight="bold",
					  va="top")
		self.lines = []
		self.cache = {}
		self.fig.canvas.draw()

	def base(self, i):
		if i not in self.cache:
			for a in self.lines:
				a.remove()
			self.lines = []
			if i >= 0:
				d = self.shown.iloc[:i + 1]
				x = d["ready_sim_s"]
				ax_f, ax_c, ax_l = self.axes
				# current values as direct labels in the right margin (never under the time cursor)
				values = []
				for tau, color, ls in zip(self.taus, SERIES, ["-", "--"]):
					y = d[f"fscore_{tau:g}"]
					self.lines += ax_f.plot(x, y, color=color, ls=ls, lw=2, marker="o", ms=4,
											label=f"F @ {tau:g} m")
					values.append((y.iloc[-1], f"F @ {tau:g} m  {y.iloc[-1]:.3f}"))
				values.sort()
				if len(values) == 2 and abs(values[1][0] - values[0][0]) < 0.06 * np.diff(ax_f.get_ylim())[0]:
					values = [(values[0][0] - 0.03 * np.diff(ax_f.get_ylim())[0], values[0][1]),
							  (values[1][0] + 0.03 * np.diff(ax_f.get_ylim())[0], values[1][1])]
				for v, s in values:
					self.lines.append(ax_f.text(1.01, v, s, transform=ax_f.get_yaxis_transform(), color=FG,
												fontsize=9, va="center"))
				self.lines.append(ax_f.legend(loc="lower left", frameon=False, labelcolor=FG, fontsize=9, ncol=2))
				self.lines += ax_c.plot(x, d["chamfer"], color=CHAMFER, lw=2, marker="o", ms=4)
				self.lines.append(ax_c.text(1.01, d["chamfer"].iloc[-1], f"{d['chamfer'].iloc[-1]:.3f} m",
											transform=ax_c.get_yaxis_transform(), color=FG, fontsize=9, va="center"))
				self.lines += ax_l.plot(x, d["latency_s"], color=LATENCY, lw=2, marker="o", ms=4)
				last = d.iloc[-1]
				self.lines.append(ax_l.text(1.01, last["latency_s"], f"{last['latency_s']:.2f} s\nRTF {last['rtf']:.2f}",
											transform=ax_l.get_yaxis_transform(), color=FG, fontsize=9,
											va="center"))
			self.cache[i] = figure_image(self.fig)
		return self.cache[i]

	def render(self, t, i) -> Image.Image:
		img = self.base(i).copy()
		top = self.axes[0].transAxes.transform((0, 1))[1]
		bottom = self.axes[2].transAxes.transform((0, 0))[1]
		x = self.axes[2].transData.transform((t, 0))[0]
		ImageDraw.Draw(img).line([(x, BOT_H - top), (x, BOT_H - bottom)], fill=ACCENT, width=2)
		return img


# ---------------------------------------------------------------- Unity video

class UnityVideo:
	def __init__(self, path, offset):
		self.reader = imageio.get_reader(path)
		meta = self.reader.get_meta_data()
		self.fps, self.offset = meta["fps"], offset
		self.n = int(meta["duration"] * self.fps)
		self.idx, self.frame = -1, None
		print(f"Unity video: {meta['size'][0]}x{meta['size'][1]}, {self.fps:.0f} fps, {meta['duration']:.1f} s")

	def get(self, t) -> Image.Image:
		want = int(np.clip(round((t + self.offset) * self.fps), 0, self.n - 1))
		if want < self.idx or want - self.idx > 2 * self.fps or self.frame is None:
			self.frame, self.idx = self.reader.get_data(want), want
		while self.idx < want:
			try:
				self.frame = self.reader.get_next_data()
				self.idx += 1
			except (IndexError, StopIteration):
				break
		img = Image.fromarray(self.frame)
		target = PANEL_W / TOP_H
		w, h = img.size
		if w / h > target:
			cw = int(round(h * target)); img = img.crop(((w - cw) // 2, 0, (w - cw) // 2 + cw, h))
		else:
			ch = int(round(w / target)); img = img.crop((0, (h - ch) // 2, w, (h - ch) // 2 + ch))
		return img.resize((PANEL_W, TOP_H), Image.LANCZOS)


# ---------------------------------------------------------------- frames

def colorbar(img: Image.Image, clim):
	d = ImageDraw.Draw(img)
	x0, y0, w, h = PANEL_W - 48, 125, 12, 200
	lut = (colormaps[CMAP](np.linspace(0, 1, h))[:, :3] * 255).astype(np.uint8)
	img.paste(Image.fromarray(np.repeat(lut[:, None, :], w, axis=1)), (x0, y0))
	d.rectangle([x0, y0, x0 + w, y0 + h], outline=MUTED)
	d.text((x0 + w, y0 - 10), "depth [m]", fill=FG, font=font(12), anchor="rb")
	for frac in (0, 0.5, 1):
		v = clim[0] + frac * (clim[1] - clim[0])
		d.text((x0 - 4, y0 + frac * h), f"{v:.1f}", fill=MUTED, font=font(11), anchor="rm")


def panel_text(d, x, y, title, lines):
	d.text((x + 12, y + 10), title, fill=FG, font=font(16, True))
	for i, line in enumerate(lines):
		d.text((x + 12, y + 34 + 20 * i), line, fill=FG, font=font(14))


def chrome(canvas: Image.Image, footer: str, t=None):
	d = ImageDraw.Draw(canvas)
	d.line([(0, TITLE_H - 1), (W, TITLE_H - 1)], fill=GRID)
	d.line([(0, H - FOOT_H), (W, H - FOOT_H)], fill=GRID)
	d.text((16, TITLE_H / 2), TITLE, fill=FG, font=font(19, True), anchor="lm")
	if t is not None:
		d.text((W - 16, TITLE_H / 2), f"simulation time {t:5.1f} s", fill=MUTED, font=font(15), anchor="rm")
	d.text((16, H - FOOT_H / 2), footer, fill=MUTED, font=font(13), anchor="lm")


def metric_lines(row, taus):
	lines = [f"P / R / F @ {tau:g} m:  {row[f'precision_{tau:g}']:.3f} / {row[f'recall_{tau:g}']:.3f} / "
			 f"{row[f'fscore_{tau:g}']:.3f}" for tau in taus]
	return lines + [f"Chamfer:  {row['chamfer']:.3f} m"]


def title_card(footer, taus, every, rate, zexag, run):
	img = Image.new("RGB", (W, H), BG)
	d = ImageDraw.Draw(img)
	d.text((W / 2, 300), TITLE, fill=FG, font=font(36, True), anchor="mm")
	d.text((W / 2, 350), f"run {os.path.basename(os.path.normpath(run))}", fill=MUTED, font=font(18), anchor="mm")
	legend = [
		"Top: Unity view | acquired point cloud | online Poisson mesh (same camera, vertical exaggeration "
		f"{zexag:g}x)",
		f"Point colour = depth for the last {every} pings (current update window); grey = earlier pings, "
		"kept in the voxel map",
		"White: newest ping, boat position, time cursor",
		"Bottom: plan view (GT hillshade, area covered by the mesh, boat track, box rebuilt at the last update) "
		"| metrics and latency per update",
		f"Metrics vs GT: precision / recall / F-score at tau = {' and '.join(f'{t:g}' for t in taus)} m, "
		"Chamfer distance [m]",
		"Clock: simulation time; mesh delay = time the mesh is ready - time of its newest data"
		+ (f" (simulation ran at {rate:.2f}x wall clock; latency in wall-clock s)" if abs(rate - 1) > 0.05 else ""),
	]
	for i, line in enumerate(legend):
		d.text((W / 2, 450 + 36 * i), line, fill=FG, font=font(18), anchor="mm")
	chrome(img, footer)
	return img


def summary_card(last_frame: Image.Image, rows: pd.DataFrame, shown: pd.DataFrame, taus, footer):
	img = Image.blend(last_frame, Image.new("RGB", (W, H), BG), 0.78)
	d = ImageDraw.Draw(img)
	x0, y0, w = W / 2 - 470, 170, 940
	d.rectangle([x0, y0, x0 + w, y0 + 700], fill=BG, outline=GRID)
	d.text((x0 + 30, y0 + 28), "Summary", fill=FG, font=font(26, True))
	d.text((x0 + 30, y0 + 66), f"Quality vs GT over {len(shown)} updates (mean ± std) and of the final mesh",
		   fill=MUTED, font=font(15))
	cols = [x0 + 30, x0 + 480, x0 + 740]
	y = y0 + 110
	d.text((cols[1], y), "mean ± std", fill=MUTED, font=font(15, True)); d.text((cols[2], y), "final", fill=MUTED,
																				font=font(15, True))
	final = shown.iloc[-1]
	entries = [(f"{m.capitalize()} @ {tau:g} m", f"{m}_{tau:g}", 3) for tau in taus for m in ("precision", "recall",
																							  "fscore")]
	entries.append(("Chamfer [m]", "chamfer", 3))
	for label, col, dec in entries:
		y += 30
		d.text((cols[0], y), label.replace("Fscore", "F-score"), fill=FG, font=font(16))
		d.text((cols[1], y), f"{shown[col].mean():.{dec}f} ± {shown[col].std():.{dec}f}", fill=FG, font=font(16))
		d.text((cols[2], y), f"{final[col]:.{dec}f}", fill=FG, font=font(16))
	full = rows[rows["full_batch"]]
	y += 55
	d.text((cols[0], y), f"Timing over {len(full)} full updates (final partial batch excluded)", fill=MUTED,
		   font=font(15))
	y += 10
	for label, col in [("Latency [s] (wall clock)", "latency_s"), ("RTF", "rtf"), ("Mesh delay [s]", "mesh_delay_s")]:
		y += 30
		d.text((cols[0], y), label, fill=FG, font=font(16))
		d.text((cols[1], y), f"mean {full[col].mean():.2f}", fill=FG, font=font(16))
		d.text((cols[2], y), f"p95 {np.percentile(full[col], 95):.2f}   max {full[col].max():.2f}", fill=FG,
			   font=font(16))
	chrome(img, footer)
	return img


def main():
	p = argparse.ArgumentParser(description="Result video of an online reconstruction session")
	p.add_argument("--run", required=True, help="online_reconstruction.py run folder")
	p.add_argument("--csv", default=None, help="Unity CSV (default: the one in <run>/config.txt)")
	p.add_argument("--unity-video", default=None, help="mp4 from the Unity Recorder (without: black panel)")
	p.add_argument("--out", default=None, help="default: output/video/video_<date_time>.mp4")
	p.add_argument("--preview", type=float, nargs="*", default=None,
				   help="only write PNG frames at these fractions of the session (default 0.1 0.5 0.95)")
	p.add_argument("--gt", default="data/real/patch_malamocco")
	p.add_argument("--terrain-position", type=float, nargs=2, default=[-128.0, -128.0], metavar=("X", "Z"))
	p.add_argument("--taus", type=float, nargs=2, default=[0.2, 0.5])
	p.add_argument("--fps", type=int, default=30)
	p.add_argument("--zexag", type=float, default=2.0, help="vertical exaggeration of the 3D panels")
	p.add_argument("--t-offset", type=float, default=0.0, help="seconds added to the Unity video time")
	p.add_argument("--max-points-per-ping", type=int, default=8000)
	p.add_argument("--point-size", type=float, default=2.0)
	p.add_argument("--title-s", type=float, default=2.0)
	p.add_argument("--end-s", type=float, default=3.0)
	a = p.parse_args()

	config = yaml.safe_load(open(os.path.join(a.run, "config.txt")))
	csv_path = a.csv or config["csv"]
	every = int(config["online"]["update_every_pings"])
	meta = json.load(open(os.path.join(a.gt, "meta.json")))

	pings = load_pings(csv_path, a.zexag, a.max_points_per_ping)
	rows, rate, ping_dt = update_table(a.run, pings, a.taus, every)
	cols = ["update", "first_ping", "last_ping", "elapsed_s", "ready_sim_s", "mesh_delay_s", "latency_s", "budget_s",
			"rtf", "full_batch", "chamfer"] + [f"{m}_{t:g}" for t in a.taus for m in ("precision", "recall", "fscore")]
	rows[cols].to_csv(os.path.join(a.run, "metrics_per_update.csv"), index=False, float_format="%.3f")
	snapshots = {int(re.search(r"mesh_(\d+)\.ply$", f).group(1)): f
				 for f in glob.glob(os.path.join(a.run, "meshes", "mesh_*.ply"))}
	shown = rows[rows["update"].isin(list(snapshots))].sort_values("ready_sim_s").reset_index(drop=True)
	if shown.empty:
		raise SystemExit(f"No mesh snapshots in {a.run}/meshes")
	print(f"{len(pings)} pings, {len(shown)} updates with a snapshot, simulation/wall clock {rate:.2f}, "
		  f"saved {os.path.join(a.run, 'metrics_per_update.csv')}")

	footer = (f"Sensor: VLP-16 (proxy for multibeam sonar), sigma 0.05 m | voxel "
			  f"{config['preprocessing']['downsampling']['voxel_size']:g} m, Poisson depth {config['poisson']['depth']}, "
			  f"window {every * ping_dt:g} s ({every} pings) | GT: Madricardo et al. 2017, MGDS doi:10.1594/IEDA/323605, "
			  f"{meta['cell_m']:g} m DTM")

	final = pv.read(snapshots[int(shown["update"].iloc[-1])])
	v = np.asarray(final.points)
	disp = to_disp(v[:, 0], v[:, 1], v[:, 2], a.zexag)
	content = disp[np.random.default_rng(0).choice(len(disp), min(len(disp), 20000), replace=False)]
	depth_all = np.concatenate([p["depth"] for p in pings])
	clim = tuple(np.percentile(depth_all, [2, 98]))
	t_end = max(pings[-1]["t"], shown["ready_sim_s"].max()) + 1.0

	cloud = CloudScene(content, pings, every, clim, a.point_size)
	mesh = MeshScene(content, snapshots, a.zexag)
	plan = PlanView(a.gt, a.terrain_position, rows, snapshots, pings)
	charts = Charts(shown, a.taus, t_end, every, ping_dt)
	unity = UnityVideo(a.unity_video, a.t_offset) if a.unity_video else None
	times = np.array([p["t"] for p in pings])
	cum_points = np.cumsum([p["n"] for p in pings])
	ready = shown["ready_sim_s"].to_numpy()

	def frame(t) -> Image.Image:
		canvas = Image.new("RGB", (W, H), BG)
		i1 = int(np.searchsorted(times, t, side="right")) - 1
		j = int(np.searchsorted(ready, t, side="right")) - 1
		row = shown.iloc[j] if j >= 0 else None
		k = int(row["update"]) if row is not None else None

		canvas.paste(unity.get(t) if unity else Image.new("RGB", (PANEL_W, TOP_H), "black"), (0, TITLE_H))
		cloud.set_time(t)
		pc = cloud.render(); colorbar(pc, clim)
		canvas.paste(pc, (PANEL_W, TITLE_H))
		mesh.set_update(k)
		canvas.paste(mesh.render(), (2 * PANEL_W, TITLE_H))
		canvas.paste(plan.render(t, k), (0, BOT_Y))
		canvas.paste(charts.render(t, j), (W // 2, BOT_Y))

		d = ImageDraw.Draw(canvas)
		for x in (PANEL_W, 2 * PANEL_W):
			d.line([(x, TITLE_H), (x, BOT_Y)], fill=GRID)
		d.line([(0, BOT_Y), (W, BOT_Y)], fill=GRID)
		d.line([(W // 2, BOT_Y), (W // 2, H - FOOT_H)], fill=GRID)
		panel_text(d, 0, TITLE_H, "Unity: boat + VLP-16", [f"t = {t:5.1f} s"])
		n_pts = int(cum_points[i1]) if i1 >= 0 else 0
		panel_text(d, PANEL_W, TITLE_H, "Acquired point cloud", [f"t = {t:5.1f} s   |   {n_pts:,} points"])
		if row is not None:
			info = [f"update {k}   |   data up to {row['elapsed_s']:.1f} s   |   mesh delay {row['mesh_delay_s']:.1f} s"]
		else:
			info = ["waiting for the first update"]
		panel_text(d, 2 * PANEL_W, TITLE_H, "Online reconstructed mesh (Poisson)", info)
		if row is not None:
			for n, line in enumerate(metric_lines(row, a.taus)):
				d.text((2 * PANEL_W + 12, TITLE_H + TOP_H - 76 + 22 * n), line, fill=FG, font=font(14))
		for x in (PANEL_W, 2 * PANEL_W):
			d.text((x + PANEL_W - 12, TITLE_H + TOP_H - 12), f"vert. exag. {a.zexag:g}x", fill=MUTED, font=font(12),
				   anchor="rb")
		chrome(canvas, footer, t)
		return canvas

	if a.preview is not None:
		fractions = a.preview or [0.1, 0.5, 0.95]
		folder = os.path.join("output", "video", f"preview_{os.path.basename(os.path.normpath(a.run))}")
		os.makedirs(folder, exist_ok=True)
		for f in fractions:
			path = os.path.join(folder, f"frame_{int(round(f * 100)):02d}.png")
			frame(f * t_end).save(path)
			print(f"saved {path}")
		title_card(footer, a.taus, every, rate, a.zexag, a.run).save(os.path.join(folder, "title.png"))
		summary_card(frame(t_end), rows, shown, a.taus, footer).save(os.path.join(folder, "summary.png"))
		print(f"saved {folder}/title.png and summary.png")
		return

	out = a.out or os.path.join("output", "video", f"video_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
	os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
	writer = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=None, pixelformat="yuv420p",
								macro_block_size=1, output_params=["-crf", "18", "-preset", "slow"])
	card = np.asarray(title_card(footer, a.taus, every, rate, a.zexag, a.run))
	for _ in range(int(a.title_s * a.fps)):
		writer.append_data(card)
	ts = np.arange(0, t_end, 1 / a.fps)
	start = time.time()
	for n, t in enumerate(ts):
		last = frame(t)
		writer.append_data(np.asarray(last))
		if n % (a.fps * 10) == 0:
			print(f"  t = {t:5.1f} / {t_end:.1f} s  ({time.time() - start:.0f} s elapsed)", flush=True)
	end = np.asarray(summary_card(last, rows, shown, a.taus, footer))
	for _ in range(int(a.end_s * a.fps)):
		writer.append_data(end)
	writer.close()
	print(f"Saved {out}")


if __name__ == "__main__":
	main()
