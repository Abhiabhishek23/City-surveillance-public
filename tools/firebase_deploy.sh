#!/usr/bin/env bash
# Simple deploy script using firebase-tools and service-account authentication
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${FIREBASE_CREDENTIALS_PATH:-}" ]; then
  echo "Set FIREBASE_CREDENTIALS_PATH to your service-account json"
  exit 1
fi
export GOOGLE_APPLICATION_CREDENTIALS="$FIREBASE_CREDENTIALS_PATH"
cd "$ROOT/dashboard"
npm install
npm run build
cd "$ROOT"
# Use firebase-tools with service account
npx firebase-tools deploy --only hosting --credential "$FIREBASE_CREDENTIALS_PATH" --project $(node -e "console.log(require('./.firebaserc').projects.default)")
