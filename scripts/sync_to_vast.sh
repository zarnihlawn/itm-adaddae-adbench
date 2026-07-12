#!/usr/bin/env bash
# Push this repo (AdaDDAE) to a Vast.ai instance. ADBench stays on the instance separately.
# Usage: bash scripts/sync_to_vast.sh <ssh-host> [remote_project_dir]
# Example: bash scripts/sync_to_vast.sh root@ssh1.vast.ai:12345 /workspace/ITM/project
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh-host> [remote_project_dir]"
  echo "Example: $0 root@ssh1.vast.ai:12345 /workspace/ITM/project"
  exit 1
fi

HOST="$1"
REMOTE_PROJECT="${2:-/workspace/ITM/project}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Syncing ${REPO_ROOT}/ -> ${HOST}:${REMOTE_PROJECT}/"
rsync -avz --progress \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'results/logs' \
  "${REPO_ROOT}/" "${HOST}:${REMOTE_PROJECT}/"

echo "Done. On the instance (ADBench must already be at ../ADBench/adbench/datasets):"
echo "  cd ${REMOTE_PROJECT} && bash scripts/vast_smoke.sh"
