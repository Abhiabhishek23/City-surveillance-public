# City Surveillance (Public) – Clone & Run

This repository provides a FastAPI backend, a Client UI, and an Admin UI to explore urban monitoring flows (YOLO + CLIP ready, optional).

This public version ships without secrets or large files. Follow the steps below to run locally.

## Quickstart (macOS/Linux)
```bash
./run.sh setup
./run.sh start
./run.sh open
```
- Landing page: `http://127.0.0.1:8000/welcome`
- Client UI: `http://127.0.0.1:8000/client_ui`
- Admin UI: `http://127.0.0.1:8000/admin`

## Requirements
- Python 3.10+ (recommend 3.11+)
- macOS/Linux shell (zsh/bash)
- Optional: FFmpeg, Node.js (if you plan to rebuild the Admin SPA)

## Environment setup
1) Create a local `.env` for the backend (not committed):
```bash
cp backend/.env.example backend/.env
```
2) For a no-auth, no-push dev run, ensure:
```
ENV=development
DEBUG=true
ALLOW_DEV_NO_AUTH=1
```
3) To enable push notifications later, set one of the following in `backend/.env`:
```
FIREBASE_CREDENTIALS_FILE=/absolute/path/to/service-account.json
# or
FCM_SERVER_KEY=your-legacy-http-key
```
(Do not commit real keys.)

## VS Code tasks (optional)
- Command Palette → Run Task →
  - "Setup (run.sh)" → venv + install
  - "Start Backend (Uvicorn)" → start backend server
  - "Start Dashboard (React)" → run dashboard dev server (optional)
  - "Open Welcome" → open landing page
  - "Stop (run.sh)", "Logs (run.sh)" (via terminal)
  
Debug tip: use the Python debugger to run `backend/main.py` if you need breakpoints.

## Processing a video
```bash
curl -F "file=@/absolute/path/to/video.mp4" \
  http://127.0.0.1:8000/process_video
# Then
curl http://127.0.0.1:8000/job_status/<job_id>
```
Alerts appear in `client_alerts` and via `ws://127.0.0.1:8000/ws/alerts`.

## What’s excluded from this public repo
- Secrets (service account JSON, `.env` values)
- Large binaries (videos, model weights)
- Local databases and generated uploads

You’ll need to add your own Firebase keys locally to test push notifications and provide your own media and model weights for full inference.

## Troubleshooting
- Port busy? `./run.sh stop` then start again, or change PORT in `backend/.env`.
- Missing deps? Re-run `./run.sh setup` (Python 3.10+).
- Admin auth in dev: set `ADMIN_BYPASS_SECRET` and use `Authorization: Bypass <secret>` in the Admin UI.

## Secrets Setup Guide (bring your own keys)
This project runs without secrets, but to enable push notifications and stable signing, fetch your own credentials and set them in `backend/.env`.

1) Create/edit your env file
```bash
cp backend/.env.example backend/.env
```

2) Stable signed URLs (recommended)
```ini
SECRET_KEY=<any-long-random-string>
```

3) Firebase Admin SDK (recommended for server-side push)
- Go to Firebase Console → Settings → Service accounts → Generate new private key.
- Save the JSON file OUTSIDE the repo (or ensure it is .gitignored).
- Point the backend to it:
```ini
FIREBASE_CREDENTIALS_PATH=/absolute/path/to/service-account.json
```

4) Legacy FCM HTTP key (alternative to Admin SDK)
- Firebase Console → Cloud Messaging → Cloud Messaging API (Legacy) → Server key.
- Add to `backend/.env`:
```ini
FCM_SERVER_KEY=<your-server-key>
```

5) Web Push (optional for browser notifications)
- Firebase Console → Cloud Messaging → Web configuration → Generate key pair.
- Add to `backend/.env`:
```ini
VAPID_PUBLIC_KEY=<your-public-vapid-key>
VAPID_PRIVATE_KEY=<your-private-vapid-key>
```

6) Redis (optional)
- If you have Redis, it will be used automatically. Otherwise, the app falls back to in-memory.
```ini
REDIS_HOST=localhost
```

7) Verify setup
```bash
./run.sh start
curl http://127.0.0.1:8000/auth/debug
curl http://127.0.0.1:8000/fcm_vapid_public_key
```
You should see your Firebase project info and whether VAPID keys are set.

## Deploy Dashboard to Firebase (manual)
- This repo includes an Actions workflow for Firebase Hosting but it’s manual-only by default.
- To enable deploys:
  1. Add repo secrets `FIREBASE_TOKEN` and `FIREBASE_PROJECT_ID`.
  2. Go to GitHub → Actions → "Build and Deploy Dashboard to Firebase Hosting" → Run workflow.
  3. If you prefer auto-deploy on push, we can re-enable it with a guarded condition checking secrets.

## License
For demo/research use. Ensure you have rights to any datasets/models you add.