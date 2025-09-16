#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DEST_DIR="${1:-$SRC_DIR/../City-surveillance-public}"

echo "[public] Source: $SRC_DIR"
echo "[public] Dest:   $DEST_DIR"

mkdir -p "$DEST_DIR"

# rsync copy while excluding secrets, venvs, caches, large/local artifacts
RSYNC_EXCLUDES=(
  ".git/"
  ".DS_Store"
  "*/__pycache__/"
  "*.pyc"
  "*.pyo"
  "*.pyd"
  ".venv/"
  "venv/"
  "backend/.venv/"
  "backend/venv/"
  "backend/.env"
  "*firebase-adminsdk*.json"
  "*service-account*.json"
  "*.db"
  "test.db"
  "backend/alerts.db"
  "backend/test.db"
  "video_uploads/"
  "backend_uploads/"
  "edge_uploads/"
  "runs/"
  "logs/"
  # Heavy datasets/training outputs
  "dist/"
  "dataset/"
  "Indian_Urban_Dataset*/"
  "kaggle_build/"
  "kaggle/working/"
  "models/"
  "Train/"
  "*.mp4"
  "*.mov"
  "*.avi"
  "*.zip"
  "*.pt"
  "*.pth"
  "*.onnx"
)

RSYNC_ARGS=(-a --delete)
for ex in "${RSYNC_EXCLUDES[@]}"; do RSYNC_ARGS+=(--exclude "$ex"); done

echo "[public] Copying sanitized repository..."
rsync "${RSYNC_ARGS[@]}" "$SRC_DIR/" "$DEST_DIR/"

# Replace README with public version if present
if [[ -f "$SRC_DIR/README_PUBLIC.md" ]]; then
  cp "$SRC_DIR/README_PUBLIC.md" "$DEST_DIR/README.md"
fi

# Ensure ease-of-use files are present
mkdir -p "$DEST_DIR/.vscode"
if [[ -f "$SRC_DIR/.vscode/tasks.json" ]]; then
  cp "$SRC_DIR/.vscode/tasks.json" "$DEST_DIR/.vscode/tasks.json"
fi
if [[ -f "$SRC_DIR/.vscode/launch.json" ]]; then
  cp "$SRC_DIR/.vscode/launch.json" "$DEST_DIR/.vscode/launch.json"
fi
if [[ -f "$SRC_DIR/run.sh" ]]; then
  cp "$SRC_DIR/run.sh" "$DEST_DIR/run.sh"
  chmod +x "$DEST_DIR/run.sh"
fi

# Explicitly include Test samples (videos) if present, overriding global *.mp4 excludes
if [[ -d "$SRC_DIR/Test samples" ]]; then
  echo "[public] Including Test samples (may be large)..."
  mkdir -p "$DEST_DIR/Test samples"
  # Copy all files from Test samples regardless of extension
  rsync -a "$SRC_DIR/Test samples/" "$DEST_DIR/Test samples/"
fi

cat <<EOF

[public] Done. Review the public folder:
  $DEST_DIR

Next steps:
  cd "$DEST_DIR"
  git init
  git add .
  git commit -m "Public release: sanitized"
  # git remote add origin <your-remote-url>
  # git push -u origin main

Remember to store your Firebase credentials outside the repo and instruct users to
copy backend/.env.example to backend/.env and fill their own keys.

If your public repo should include example media or weights, initialize Git LFS first:
  git lfs install
  git lfs track "*.mp4" "*.mov" "*.avi" "*.zip" "*.pt" "*.onnx"
  git add .gitattributes
  git commit -m "chore(lfs): track large binaries"
EOF
