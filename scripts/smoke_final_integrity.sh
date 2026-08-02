#!/usr/bin/env bash
# Phase-1 smoke integrity for frozen AdaDDAE final recipe.
# Local CPU: bash scripts/smoke_final_integrity.sh
# Vast GPU:  bash scripts/smoke_final_integrity.sh 16gb
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

HW_ARG=()
if [[ "${1:-}" != "" ]]; then
  HW_ARG=(--hardware "$1")
fi

echo "=== assert final config schema (smoke yaml) ==="
"$PYTHON" scripts/assert_final_config.py \
  --config configs/adadae_final_smoke.yaml \
  --allow-nonfinal-run-id

echo "=== run smoke (3 datasets × 2 settings × 2 seeds) ==="
"$PYTHON" scripts/smoke_final_integrity.py \
  --config configs/adadae_final_smoke.yaml \
  "${HW_ARG[@]}"

echo ""
echo "Smoke passed. For full Phase-1 570 on Vast:"
echo "  bash scripts/run_adadae_final_protocol.sh all 16gb"
