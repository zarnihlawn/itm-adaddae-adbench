#!/usr/bin/env bash
# Reproduce primary integrity Table 1 (after Vast 570 results exist).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON=python3

HARDWARE="${1:-}"
HW=()
[[ -n "$HARDWARE" ]] && HW=(--hardware "$HARDWARE")

echo "=== config assert ==="
"$PYTHON" scripts/assert_final_config.py --config configs/adadae_final.yaml

echo "=== smoke (cheap) ==="
bash scripts/smoke_final_integrity.sh "${HARDWARE:-}"

echo "=== integrity audit ==="
"$PYTHON" scripts/integrity_audit.py --config configs/adadae_final.yaml \
  --completed results/adadae_final/metrics/completed.json \
  --out results/adadae_final/thesis/integrity_audit.md --quick || true

if [[ -f results/adadae_final/metrics/completed.json && -f results/ddae_baseline_valstop/metrics/completed.json ]]; then
  echo "=== compare + gates + stats ==="
  mkdir -p results/adadae_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae_final/thesis
  "$PYTHON" scripts/validate_gates.py --integrity \
    --completed results/adadae_final/metrics/completed.json \
    --compare results/adadae_final/thesis/compare_to_ddae.json \
    --logs-dir results/adadae_final/logs \
    --out results/adadae_final/thesis/integrity_gates.json || true
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae_final/thesis
else
  echo "NOTE: full 570 completed.json missing — run on Vast:"
  echo "  bash scripts/run_adadae_final_protocol.sh all 16gb"
fi

echo "repro_final done"
