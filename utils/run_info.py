"""
Informazioni di contorno di una run di evaluate.py, che dopo non si possono ricostruire:
  - ambiente(): macchina, GPU, versioni delle librerie
  - stato_git(): commit, ramo, modifiche non committate (+ diff)
  - TeeLog: copia in un file tutto quello che finisce sul terminale (anche dei sottoprocessi)
  - MonitorRisorse: picco di RAM (processo + figli, es. PPSurf) e di VRAM durante un blocco di codice

Tutto leggero: il monitor campiona RAM e VRAM una volta al secondo in un thread a parte.
"""

import os
import platform
import socket
import subprocess
import sys
import threading

try:
	import psutil
except ImportError:  # senza psutil la RAM non viene misurata
	psutil = None


def _esegui(cmd: list, **kwargs) -> str:
	"""stdout di un comando, "" se fallisce o non esiste."""
	try:
		return subprocess.run(cmd, capture_output=True, text=True, timeout=60, **kwargs).stdout.strip()
	except (OSError, subprocess.SubprocessError):
		return ""


def ambiente(python_ppsurf: str = None) -> dict:
	"""Macchina e versioni. python_ppsurf: interprete dell'ambiente di PPSurf, per la versione di PyTorch."""
	cpu = ""
	if os.path.exists("/proc/cpuinfo"):
		with open("/proc/cpuinfo") as f:
			cpu = next((r.split(":", 1)[1].strip() for r in f if r.startswith("model name")), "")
	info = {
		"hostname": socket.gethostname(),
		"sistema": platform.platform(),
		"cpu": cpu,
		"n_cpu": os.cpu_count(),
		"ram_totale_gb": round(psutil.virtual_memory().total / 2**30, 1) if psutil else None,
		"gpu": _esegui(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]) or None,
		"python": platform.python_version(),
	}
	for modulo in ("open3d", "numpy", "pandas"):
		try:
			info[modulo] = __import__(modulo).__version__
		except ImportError:
			pass
	if python_ppsurf:
		info["pytorch_ppsurf"] = _esegui([python_ppsurf, "-c",
			"import torch; print(torch.__version__, 'CUDA', torch.version.cuda)"]) or None
	return info


def stato_git(cartella: str = ".") -> tuple:
	"""(info, diff): commit, ramo e file modificati; diff = git diff rispetto al commit ("" se pulito)."""
	commit = _esegui(["git", "rev-parse", "HEAD"], cwd=cartella)
	if not commit:
		return {"git": "non disponibile"}, ""
	modificati = _esegui(["git", "status", "--porcelain"], cwd=cartella).splitlines()
	info = {
		"commit": commit,
		"ramo": _esegui(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cartella),
		"modifiche_non_committate": [r.strip() for r in modificati] or "nessuna",
	}
	return info, _esegui(["git", "diff", "HEAD"], cwd=cartella)


class TeeLog:
	"""Duplica stdout e stderr (a livello di file descriptor, quindi anche l'output dei sottoprocessi
	e delle librerie C++ come Open3D) sul terminale e in un file."""

	def __init__(self, percorso: str):
		self.percorso = percorso

	def __enter__(self):
		sys.stdout.flush(); sys.stderr.flush()
		self.file = open(self.percorso, "ab")
		self.originale_out, self.originale_err = os.dup(1), os.dup(2)
		lettura, scrittura = os.pipe()
		os.dup2(scrittura, 1); os.dup2(scrittura, 2)
		os.close(scrittura)
		self.lettura = lettura
		self.thread = threading.Thread(target=self._copia, daemon=True)
		self.thread.start()
		sys.stdout.reconfigure(line_buffering=True)  # su una pipe Python bufferizza a blocchi
		return self

	def _copia(self):
		while True:
			dati = os.read(self.lettura, 65536)
			if not dati:
				break
			os.write(self.originale_out, dati)
			self.file.write(dati); self.file.flush()

	def __exit__(self, *exc):
		sys.stdout.flush(); sys.stderr.flush()
		os.dup2(self.originale_out, 1); os.dup2(self.originale_err, 2)  # chiude l'ultimo lato di scrittura
		self.thread.join(timeout=5)
		os.close(self.lettura); os.close(self.originale_out); os.close(self.originale_err)
		self.file.close()
		return False


class MonitorRisorse:
	"""Picco di RAM (processo + tutti i figli) e di VRAM (rispetto all'inizio) durante il blocco with.
	Dopo l'uscita: self.ram_picco_mb, self.vram_picco_mb (None se non misurabili)."""

	def __init__(self, intervallo_s: float = 1.0):
		self.intervallo = intervallo_s
		self.ram_picco_mb = None
		self.vram_picco_mb = None

	@staticmethod
	def _ram_mb():
		if psutil is None:
			return None
		processo = psutil.Process()
		totale = 0
		for p in [processo] + processo.children(recursive=True):
			try:
				totale += p.memory_info().rss
			except psutil.Error:  # figlio terminato nel frattempo
				pass
		return totale / 2**20

	@staticmethod
	def _vram_mb():
		uscita = _esegui(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
		try:
			return float(uscita.splitlines()[0])
		except (IndexError, ValueError):
			return None

	def _campiona(self):
		ram, vram = self._ram_mb(), self._vram_mb()
		if ram is not None:
			self.ram_picco_mb = max(self.ram_picco_mb or 0.0, ram)
		if vram is not None and self._vram_base is not None:
			self.vram_picco_mb = max(self.vram_picco_mb or 0.0, vram - self._vram_base)

	def _ciclo(self):
		while not self._stop.wait(self.intervallo):
			self._campiona()

	def __enter__(self):
		self._vram_base = self._vram_mb()
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._ciclo, daemon=True)
		self._thread.start()
		return self

	def __exit__(self, *exc):
		self._stop.set()
		self._thread.join()
		self._campiona()
		return False
