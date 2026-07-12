#!/usr/bin/env bash
# Pull results from a Vast.ai instance before terminating it.
# Usage: bash scripts/sync_results_from_vast.sh <ssh-host> [remote_dir] [local_dir]
# Example: bash scripts/sync_results_from_vast.sh root@ssh1.vast.ai:12345
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <ssh-host> [remote_project_dir] [local_project_dir]"
  echo "Example: $0 root@ssh1.vast.ai:12345 /workspace/ITM/project"
  exit 1
fi

HOST="$1"
REMOTE_PROJECT="${2:-/workspace/ITM/project}"
LOCAL_PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -ge 3 ]]; then
  LOCAL_PROJECT="$3"
fi

mkdir -p "${LOCAL_PROJECT}/results"

echo "Pulling results ${HOST}:${REMOTE_PROJECT}/results/ -> ${LOCAL_PROJECT}/results/"
rsync -avz --progress \
  "${HOST}:${REMOTE_PROJECT}/results/" "${LOCAL_PROJECT}/results/"

if [[ -f "${LOCAL_PROJECT}/results/metrics/completed.json" ]]; then
  N="$(python3 - <<PY
import json
from pathlib import Path
p = Path("${LOCAL_PROJECT}/results/metrics/completed.json")
d = json.loads(p.read_text())
print(len(d.get("completed", {})))
PY
)"
  echo "Resume state: ${N} jobs in completed.json"
  echo "To continue on Vast: python scripts/run_full_protocol.py --config configs/default_gpu.yaml"
else
  echo "No completed.json yet (protocol not started or empty results)."
fi

echo "Sync complete."
