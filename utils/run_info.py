"""Run metadata for evaluate.py: machine and library versions, git state, a terminal tee to a log file,
and a monitor of peak RAM and VRAM."""

import os
import platform
import socket
import subprocess
import sys
import threading

try:
	import psutil
except ImportError:  # without psutil RAM is not measured
	psutil = None


def _run(cmd: list, **kwargs) -> str:
	"""stdout of a command, "" if it fails or does not exist."""
	try:
		return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kwargs).stdout.strip()
	except (OSError, subprocess.SubprocessError):
		return ""


def environment(ppsurf_python: str = None) -> dict:
	"""Machine and versions. ppsurf_python: interpreter of the PPSurf environment, for the PyTorch version."""
	cpu = ""
	if os.path.exists("/proc/cpuinfo"):
		with open("/proc/cpuinfo") as f:
			cpu = next((r.split(":", 1)[1].strip() for r in f if r.startswith("model name")), "")
	info = {
		"hostname": socket.gethostname(),
		"system": platform.platform(),
		"cpu": cpu,
		"n_cpu": os.cpu_count(),
		"total_ram_gb": round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
		"gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) or None,
		"python": platform.python_version(),
	}
	for module in ("open3d", "numpy", "pandas"):
		try:
			info[module] = __import__(module).__version__
		except ImportError:
			pass
	if ppsurf_python:
		info["pytorch_ppsurf"] = _run([ppsurf_python, "-c",
			"import torch; print(torch.__version__, 'CUDA', torch.version.cuda)"]) or None
	return info


def git_state(folder: str = ".") -> tuple:
	"""(info, diff): commit, branch and modified files; diff = git diff against the commit ("" if clean)."""
	commit = _run(["git", "rev-parse", "HEAD"], cwd=folder)
	if not commit:
		return {"git": "not available"}, ""
	modified = _run(["git", "status", "--porcelain"], cwd=folder).splitlines()
	info = {
		"commit": commit,
		"branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=folder),
		"uncommitted_changes": [r.strip() for r in modified] or "none",
	}
	return info, _run(["git", "diff", "HEAD"], cwd=folder)


class TeeLog:
	"""Duplicates stdout and stderr at file-descriptor level (so also subprocesses and C++ libraries
	such as Open3D) to the terminal and to a file."""

	def __init__(self, path: str):
		self.path = path

	def __enter__(self):
		sys.stdout.flush(); sys.stderr.flush()
		self.file = open(self.path, "ab")
		self.orig_out, self.orig_err = os.dup(1), os.dup(2)
		read_fd, write_fd = os.pipe()
		os.dup2(write_fd, 1); os.dup2(write_fd, 2)
		os.close(write_fd)
		self.read_fd = read_fd
		self.thread = threading.Thread(target=self._copy, daemon=True)
		self.thread.start()
		sys.stdout.reconfigure(line_buffering=True)  # on a pipe Python buffers in blocks
		return self

	def _copy(self):
		while True:
			data = os.read(self.read_fd, 65536)
			if not data:
				break
			os.write(self.orig_out, data)
			self.file.write(data); self.file.flush()

	def __exit__(self, *exc):
		sys.stdout.flush(); sys.stderr.flush()
		os.dup2(self.orig_out, 1); os.dup2(self.orig_err, 2)  # closes the last write end
		self.thread.join(timeout=5)
		os.close(self.read_fd); os.close(self.orig_out); os.close(self.orig_err)
		self.file.close()
		return False


class ResourceMonitor:
	"""Peak RAM (process + all children) and VRAM (relative to the start) during the with block.
	After exit: self.peak_ram_mb, self.peak_vram_mb (None if not measurable)."""

	def __init__(self, interval_s: float = 1.0):
		self.interval = interval_s
		self.peak_ram_mb = None
		self.peak_vram_mb = None

	@staticmethod
	def _ram_mb():
		if psutil is None:
			return None
		process = psutil.Process()
		total = 0
		for p in [process] + process.children(recursive=True):
			try:
				total += p.memory_info().rss
			except psutil.Error:  # child exited meanwhile
				pass
		return total / 2**20

	@staticmethod
	def _vram_mb():
		out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
		try:
			return float(out.splitlines()[0])
		except (IndexError, ValueError):
			return None

	def _sample(self):
		ram, vram = self._ram_mb(), self._vram_mb()
		if ram is not None:
			self.peak_ram_mb = max(self.peak_ram_mb or 0.0, ram)
		if vram is not None and self._vram_base is not None:
			self.peak_vram_mb = max(self.peak_vram_mb or 0.0, vram - self._vram_base)

	def _loop(self):
		while not self._stop.wait(self.interval):
			self._sample()

	def __enter__(self):
		self._vram_base = self._vram_mb()
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._loop, daemon=True)
		self._thread.start()
		return self

	def __exit__(self, *exc):
		self._stop.set()
		self._thread.join()
		self._sample()
		return False
