"""
Minimal smoke test to verify the backend starts and the health endpoint responds.
Usage:
  python smoke_tests.py
"""
from __future__ import annotations
import multiprocessing as mp
import time
import os
import sys


def _run_server():
	# Run uvicorn server for backend/main.py
	os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))
	from uvicorn import Server, Config
	config = Config('main:app', host='127.0.0.1', port=8000, log_level='warning')
	srv = Server(config)
	srv.run()


def main():
	p = mp.Process(target=_run_server, daemon=True)
	p.start()
	try:
		import requests
		ok = False
		for _ in range(60):
			try:
				time.sleep(0.25)
				r = requests.get('http://127.0.0.1:8000/')
				if r.status_code == 200 and r.json().get('status') == 'ok':
					ok = True
					break
			except Exception:
				pass
		print('Backend health:', 'OK' if ok else 'FAILED')
		sys.exit(0 if ok else 1)
	finally:
		p.terminate(); p.join(timeout=5)


if __name__ == '__main__':
	main()
