"""
MVP Orchestrator: Start backend (Uvicorn), dashboard (React), and AI client (GUI) in parallel.

Usage (from repo root):
  python3 urban_monitoring_deployment.py --project-root "/path/to/City-surveillance 2"

This script will:
- Ensure it runs under the local .venv Python (auto-bootstrap).
- Start backend on 127.0.0.1:8000 unless already running.
- Start dashboard on http://localhost:3000 unless already running.
- Start AI client with GUI using yolov12n.pt on Test samples/3.mp4 by default.
- Keep running, showing concise logs; Ctrl+C cleans up children it started.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
import glob


def _print(msg: str):
	print(f"[orchestrator] {msg}")


def ensure_venv(project_root: Path):
	"""Re-exec into the project .venv python if not already inside it."""
	venv_python = project_root / ".venv" / "bin" / "python"
	if not venv_python.exists():
		_print("Warning: .venv not found; proceeding with current interpreter.")
		return
	# If current python isn't the venv python, re-exec
	try:
		cur = Path(sys.executable).resolve()
		if cur != venv_python.resolve():
			_print(f"Re-exec with venv interpreter: {venv_python}")
			os.execv(str(venv_python), [str(venv_python)] + sys.argv)
	except Exception as e:
		_print(f"venv bootstrap failed ({e}); continuing with current interpreter.")


def is_port_listening(host: str, port: int, timeout: float = 0.5) -> bool:
	try:
		with socket.create_connection((host, port), timeout=timeout):
			return True
	except OSError:
		return False


def wait_for_http(url: str, timeout_s: float = 15.0) -> bool:
	try:
		import urllib.request
		start = time.time()
		while time.time() - start < timeout_s:
			try:
				with urllib.request.urlopen(url, timeout=2) as resp:
					if resp.status < 500:
						return True
			except Exception:
				time.sleep(0.5)
		return False
	except Exception:
		return False


class Proc:
	def __init__(self, name: str, popen: subprocess.Popen, owns: bool):
		self.name = name
		self.popen = popen
		self.owns = owns  # whether we started it; if False we won't try to kill


def start_backend(project_root: Path, host: str, port: int) -> Proc:
	if is_port_listening(host, port):
		_print(f"Backend already listening on {host}:{port}; skipping start.")
		return Proc("backend", popen=subprocess.Popen(["sleep", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), owns=False)

	env = os.environ.copy()
	env.setdefault("UVICORN_LOG_LEVEL", "info")
	cwd = project_root / "backend"
	venv_python = project_root / ".venv" / "bin" / "python"
	py = str(venv_python) if venv_python.exists() else sys.executable
	cmd = [py, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port)]
	_print(f"Starting backend: {' '.join(cmd)} (cwd={cwd})")
	p = subprocess.Popen(cmd, cwd=str(cwd), env=env, preexec_fn=os.setsid)
	ok = wait_for_http(f"http://{host}:{port}/", timeout_s=20)
	_print("Backend health: OK" if ok else "Backend health: TIMEOUT")
	return Proc("backend", p, owns=True)


def start_dashboard(project_root: Path, port: int) -> Proc:
	if is_port_listening("127.0.0.1", port) or is_port_listening("localhost", port):
		_print(f"Dashboard already listening on :{port}; skipping start.")
		return Proc("dashboard", popen=subprocess.Popen(["sleep", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), owns=False)

	cwd = project_root / "dashboard"
	# Prefer npm if available; react-scripts prompts are disabled by CI env
	env = os.environ.copy()
	env.setdefault("BROWSER", "none")  # don't auto-open browser tab
	cmd = ["npm", "start"]
	_print(f"Starting dashboard dev server: {' '.join(cmd)} (cwd={cwd})")
	p = subprocess.Popen(cmd, cwd=str(cwd), env=env, preexec_fn=os.setsid)
	ok = wait_for_http(f"http://localhost:{port}", timeout_s=30)
	_print("Dashboard reachable: OK" if ok else "Dashboard reachable: TIMEOUT (check terminal)")
	return Proc("dashboard", p, owns=True)


def start_client(project_root: Path, backend_url: str, model: str, source: str, camera: str, crowd_threshold: int) -> Proc:
	client_py = project_root / "kaggle" / "working" / "urban_monitoring_deployment" / "urban_monitor_deploy.py"
	if not client_py.exists():
		raise FileNotFoundError(f"Client script not found at {client_py}")
	# Run with GUI (no --no-gui) so a window appears
	venv_python = project_root / ".venv" / "bin" / "python"
	py = str(venv_python) if venv_python.exists() else sys.executable
	cmd = [
		py,
		str(client_py),
		"--model", model,
		"--backend", backend_url,
		"--source", source,
		"--camera", camera,
		"--crowd-threshold", str(crowd_threshold),
	]
	# Allow env overrides for downstream loaders
	env = os.environ.copy()
	if os.path.exists(model):
		env.setdefault("YOLO_MODEL", model)
		env.setdefault("KAGGLE_BEST_PT", model)
	_print(f"Starting AI client (GUI): {' '.join(cmd)} (cwd={project_root})")
	p = subprocess.Popen(cmd, cwd=str(project_root), env=env, preexec_fn=os.setsid)
	return Proc("client", p, owns=True)


def auto_discover_model(project_root: Path) -> Optional[str]:
	# 1) env hints
	for k in ("YOLO_MODEL", "KAGGLE_BEST_PT"):
		v = os.getenv(k)
		if v and os.path.exists(v):
			return v
	# 2) common local paths
	candidates = [
		project_root / "weights" / "best.pt",
		project_root / "Edge" / "weights" / "best.pt",
		project_root / "models" / "best_combined.pt",
	]
	for p in candidates:
		if p.exists():
			return str(p)
	# 3) latest from runs
	runs = sorted(glob.glob(str(project_root / 'runs' / 'detect' / '*' / 'weights' / 'best.pt')))
	if runs:
		return runs[-1]
	return None


def parse_args():
	ap = argparse.ArgumentParser(description="MVP Orchestrator")
	ap.add_argument("--project-root", default=str(Path(__file__).parent.resolve()), help="Path to project root")
	ap.add_argument("--backend-host", default="127.0.0.1")
	ap.add_argument("--backend-port", type=int, default=8000)
	ap.add_argument("--dashboard-port", type=int, default=3000)
	ap.add_argument("--model", default="yolov12n.pt")
	ap.add_argument("--source", default="Test samples/3.mp4")
	ap.add_argument("--camera", default="CAM_DEMO")
	ap.add_argument("--crowd-threshold", type=int, default=5)
	ap.add_argument("--no-backend", action="store_true")
	ap.add_argument("--no-dashboard", action="store_true")
	ap.add_argument("--no-client", action="store_true")
	return ap.parse_args()


def main():
	args = parse_args()
	project_root = Path(args.project_root).resolve()

	# Bootstrap into venv python if present
	ensure_venv(project_root)

	procs: List[Proc] = []

	try:
		if not args.no_backend:
			procs.append(start_backend(project_root, args.backend_host, args.backend_port))
		else:
			_print("Skipping backend by flag")

		if not args.no_dashboard:
			procs.append(start_dashboard(project_root, args.dashboard_port))
		else:
			_print("Skipping dashboard by flag")

		if not args.no_client:
			backend_url = f"http://{args.backend_host}:{args.backend_port}/alerts"
			# Respect explicit model path or filename if it exists; only auto-discover when missing
			model_arg = args.model
			if not os.path.exists(model_arg):
				best = auto_discover_model(project_root)
				if best:
					_print(f"Auto-discovered model: {best} (overriding {model_arg})")
					model_arg = best
			else:
				_print(f"Using explicit model: {model_arg}")
			procs.append(start_client(project_root, backend_url, model_arg, args.source, args.camera, args.crowd_threshold))
		else:
			_print("Skipping client by flag")

		_print("All requested services launched. Press Ctrl+C to stop.")

		# Wait on client primarily; keep orchestrator alive while children run
		while True:
			# If client exited, we still keep backend/dashboard running; break only on Ctrl+C
			time.sleep(1)
			# Optionally, print a heartbeat
			# _print("heartbeat")
	except KeyboardInterrupt:
		_print("SIGINT received; shutting down owned processes...")
	finally:
		# Terminate only processes we started (owns=True)
		for pr in procs:
			if not pr.owns:
				continue
			try:
				_print(f"Stopping {pr.name} (pid={pr.popen.pid})")
				# Send SIGINT to the whole process group
				os.killpg(os.getpgid(pr.popen.pid), signal.SIGINT)
			except Exception:
				try:
					pr.popen.terminate()
				except Exception:
					pass
		# Give them a moment to exit cleanly
		time.sleep(1.5)
		for pr in procs:
			if not pr.owns:
				continue
			if pr.popen.poll() is None:
				try:
					os.killpg(os.getpgid(pr.popen.pid), signal.SIGKILL)
				except Exception:
					try:
						pr.popen.kill()
					except Exception:
						pass
		_print("Shutdown complete.")


if __name__ == "__main__":
	main()

