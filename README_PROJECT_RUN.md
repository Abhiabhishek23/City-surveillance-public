Project quick-run and cleanup notes

What I will do next:
- Start the backend FastAPI server locally in a venv.
- Start the frontend dashboard with npm.
- Provide commands to run the urban monitoring client from `kaggle/working/urban_monitoring_deployment`.

Suggested cleanup actions (I have NOT deleted anything without explicit confirmation):
- Remove `Data/combined_dataset.zip` (832MB) if you already have `Data/combined_dataset/` extracted. It duplicates content.
- Remove the embedded virtualenv at `kaggle/working/urban_monitoring_deployment/path` (99MB). It's a local copy of venv and not needed in the repo.
- Keep a single `yolov8n.pt` at repo root and symlink from `Edge/` (done).
- Optionally remove older training run folders under `runs/detect/*` if not required.

Run steps

1) Backend (macOS):

python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

2) Dashboard:

cd dashboard
npm install
npm start

3) Urban Monitor client (local test):

python -m venv .venv
source .venv/bin/activate
pip install -r kaggle/working/urban_monitoring_deployment/requirements.txt
python kaggle/working/urban_monitoring_deployment/urban_monitor_deploy.py --model kaggle/working/urban_monitoring_deployment/urban_monitor.onnx --backend http://127.0.0.1:8000/alerts --source 0 --camera CAM01

If you want, I can proceed to run the backend start now and fix any runtime errors I encounter.

## Additional utilities added

- `backend/scripts/register_test_personnel.py` — create a test Personnel entry and register an FCM token with the running backend. Example:

```bash
python backend/scripts/register_test_personnel.py --name TestUser --token FAKE_TOKEN_123 --backend http://127.0.0.1:8000
```

- `tools/cleanup_large_files.sh` — moves large duplicate files (e.g., embedded venv, dataset zip) to `~/cs_backup` safely. Run and inspect `$HOME/cs_backup` before deleting.

- `docker-compose.prod.yml` — skeleton to run backend + dashboard in containers. Edge clients are expected to run on edge devices.

## Per-frame client logs

The urban client writes per-frame logs to `logs/client_frame_log.jsonl` (JSON Lines). Each line contains timestamp, camera_id, counts, and detections with confidences. Use this to tune `CROWD_THRESHOLD`.

## Notes on CROWD_THRESHOLD

- For local testing I temporarily exported `CROWD_THRESHOLD=1` in `run_local.sh`.
- The backend now supports a default `CROWD_THRESHOLD` and an optional `CROWD_THRESHOLD_MAP` env var (JSON) to configure per-camera thresholds, e.g.:

```bash
export CROWD_THRESHOLD=3
export CROWD_THRESHOLD_MAP='{"CAM_TEST_1":1, "CAM02":5}'
```

This allows different sensitivity per camera.

## Firebase hosting & CI/CD (brief)

1. Create Firebase project in console and enable Hosting. Add your domain in Hosting later.
2. Place your Firebase service account JSON somewhere safe on the server (do NOT commit). Example path: `/etc/keys/ujjain-firebase-service.json` and set `FIREBASE_CREDENTIALS_PATH` to that path in environment.
3. Templates added to the repo:
   - `firebase.json` (serves `dashboard/build` and rewrites to `index.html`)
   - `.firebaserc` (project aliases)
   - GitHub Actions workflow at `.github/workflows/firebase-hosting.yml` (builds `dashboard`, uses `FIREBASE_TOKEN` or `FIREBASE_PROJECT_ID` secret to deploy)

Quick deploy via token (for CI):
- Create a CI token: `firebase login:ci` (follow prompts) and add the token to GitHub Secrets as `FIREBASE_TOKEN` and your project id as `FIREBASE_PROJECT_ID`.
- Workflow will run on `push` to `main` and deploy `dashboard/build` to Firebase Hosting.

Security & env
- Do not keep private keys or service-account JSON in the repo.
- Set `FIREBASE_CREDENTIALS_PATH` in production environment for server-side Admin SDK and keep `FCM_KEY` only for fallback.

Admin UI integration
- The dashboard SPA now includes an Admin route at `/admin` (component `dashboard/src/AdminStats.js`). It fetches `/stats/aggregate` from the backend. You can control access with your site's auth.

Local dashboard dev setup note
- The dashboard `package.json` was updated to include `react-router-dom`. To install dependencies locally run inside `dashboard`:

```bash
cd dashboard
npm install
npm start
```

CI environments using the included GitHub Actions workflow should set `FIREBASE_TOKEN` and `FIREBASE_PROJECT_ID` secrets.

Service account JSON
- Place your Firebase service-account JSON file outside the repo (example `/etc/keys/ujjain-50cb1-firebase-adminsdk.json`) and set `FIREBASE_CREDENTIALS_PATH` environment variable to that path before starting the backend so server-side Firebase Admin can be used.

## Firestore security rules
- A template `firestore.rules` has been added. Key points:
  - User documents should only be readable/writable by the authenticated user.
  - Alerts should not be writable by client SDKs (server writes via Admin SDK only).
  - Enforce App Check and/or server-side IAM for server writes in production.

## Sentry monitoring (optional)
- Frontend: set `REACT_APP_SENTRY_DSN` in your environment for build-time injection.
- Backend: set `SENTRY_DSN` as environment variable before starting the server.
- Example for backend:

```bash
export SENTRY_DSN="https://abc@o0.ingest.sentry.io/000"
export FIREBASE_CREDENTIALS_PATH="/etc/keys/ujjain-50cb1-firebase-adminsdk.json"
export FCM_KEY="..."
/tmp/citysrv_test_venv/bin/uvicorn main:app --app-dir backend --host 0.0.0.0 --port 8000
```

## Performance & caching
- Build the dashboard with `npm run build` for production (creates minified bundles).
- Configure long-lived `Cache-Control` headers for static assets; Firebase Hosting sets sensible defaults and supports custom headers in `firebase.json`.
- Use hashed filenames for cache-busting (CRA does this by default).

## Deploy script
- A helper script `tools/firebase_deploy.sh` runs a build and deploy using a service-account JSON. Example usage:

```bash
export FIREBASE_CREDENTIALS_PATH="/etc/keys/ujjain-50cb1-firebase-adminsdk.json"
bash tools/firebase_deploy.sh
```

*** End of additions

## Train → Validate → Infer-and-Alert

End-to-end steps for detection fine-tuning and live alerts.

1) Train (Kaggle recommended, T4 GPU)

- Attach your YOLO dataset (must have `images/`, `labels/`, `data.yaml`).
- Also attach this project's dataset extras if needed (contains run scripts/weights).

Example inside Kaggle notebook or Terminal:

```bash
# If you attached your dataset at /kaggle/input/<yolo_slug>/data.yaml
yolo detect train \
  model=/kaggle/input/<extras_slug>/Indian_Urban_Dataset/weights/yolov12n.pt \
  data=/kaggle/input/<yolo_slug>/data.yaml \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
  project="runs/detect" name="indian-urban-y12"
```

2) Validate

```bash
yolo detect val \
  model=runs/detect/indian-urban-y12/weights/best.pt \
  data=/kaggle/input/<yolo_slug>/data.yaml \
  device=0 imgsz=640
```

3) Infer and alert (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ultralytics shapely

# Start backend in another terminal (see earlier section) then run:
python tools/infer_and_alert.py \
  --weights runs/detect/indian-urban-y12/weights/best.pt \
  --data Indian_Urban_Dataset_yolo/data.yaml \
  --source Assets/test_land.mp4 \
  --backend http://127.0.0.1:8000 \
  --camera-id CAM_TEST \
  --conf 0.25 --iou 0.45
```

Notes
- Attribute rules live in `config/attribute_rules.json` and can encode legal/illegal/conditional status per class and per-zone policies (see `config/zones.geojson`).
- Snapshots are saved under `edge_uploads/` and posted to `/alerts`; the backend enforces per-event cooldowns and FCM notifications.
- For zone-aware checks, add polygons to `config/zones.geojson` and specify properties like `{"zone_type": "no_hawker_zone"}` that match the rules file. See `README_ZONES.md` for schema, examples, and a build tool workflow.
