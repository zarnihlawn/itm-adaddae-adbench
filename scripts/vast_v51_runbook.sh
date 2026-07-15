#!/usr/bin/env bash
# Run this entire script ON Vast after git pull. Do not nest tmux.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== AdaDDAE v5.1 Vast full run (~130 GPU jobs, 8-12h) ==="
source .venv/bin/activate
bash scripts/vast_smoke.sh 12gb

# Full pipeline
USE_GATE=1 bash scripts/run_adadae_v51_protocol.sh vast-all

echo ""
echo "=== Post-run: commit and push ==="
git add results/adadae_v51_* configs/damp_model.pkl configs/damp_model.meta.json
git status
echo "Run: git commit -m 'AdaDDAE v5.1 MCE+SMC+guarded merge' && git push"
