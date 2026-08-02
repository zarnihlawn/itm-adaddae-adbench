#!/usr/bin/env bash
# Primary thesis protocol: fair DDAE val-stop 570 + AdaDDAE final 570 + integrity gates.
#
# Usage:
#   bash scripts/run_adadae_final_protocol.sh smoke [hardware]
#   bash scripts/run_adadae_final_protocol.sh ddae [hardware]     # fair baseline 570
#   bash scripts/run_adadae_final_protocol.sh final [hardware]    # AdaDDAE final 570
#   bash scripts/run_adadae_final_protocol.sh compare
#   bash scripts/run_adadae_final_protocol.sh gates
#   bash scripts/run_adadae_final_protocol.sh all [hardware]      # smoke → ddae → final → compare → gates
#
# Needs GPU for full 570 (rent Vast). Local machine without CUDA can only do smoke on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

MODE="${1:-smoke}"
HARDWARE="${2:-}"
HW_ARG=()
if [[ -n "$HARDWARE" ]]; then
  HW_ARG=(--hardware "$HARDWARE")
elif "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  HARDWARE="$("$PYTHON" scripts/detect_hardware.py 2>/dev/null | awk -F= '/^suggested_profile=/{print $2}' | sed 's/hardware_//;s/\.yaml$//' || true)"
  if [[ -n "${HARDWARE:-}" ]]; then
    HW_ARG=(--hardware "$HARDWARE")
  fi
fi

FINAL_COMPLETED="results/adadae_final/metrics/completed.json"
DDAE_COMPLETED="results/ddae_baseline_valstop/metrics/completed.json"
THESIS_DIR="results/adadae_final/thesis"

need_gpu() {
  if ! "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: CUDA required for mode '$MODE' (full 570). Rent Vast and re-run."
    echo "  Example: bash scripts/run_adadae_final_protocol.sh all 16gb"
    exit 2
  fi
}

run_smoke() {
  echo "=== Phase-1 smoke integrity ==="
  bash scripts/smoke_final_integrity.sh "${HARDWARE:-}"
}

run_ddae() {
  need_gpu
  echo "=== Fair DDAE baseline 570 (val-only early stop) ==="
  "$PYTHON" scripts/assert_final_config.py \
    --config configs/baselines_ddae_valstop.yaml \
    --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/baselines_ddae_valstop.yaml \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  echo "=== AdaDDAE final 570 (frozen recipe) ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae_final.yaml
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_final.yaml \
    "${HW_ARG[@]}"
}

run_compare() {
  echo "=== Table 1 compare (final vs fair DDAE) ==="
  mkdir -p "$THESIS_DIR"
  if [[ ! -f "$FINAL_COMPLETED" ]]; then
    echo "Missing $FINAL_COMPLETED"
    exit 1
  fi
  if [[ ! -f "$DDAE_COMPLETED" ]]; then
    echo "Missing $DDAE_COMPLETED — run ddae track first (or point backup)."
    exit 1
  fi
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed "$FINAL_COMPLETED" \
    --baseline "$DDAE_COMPLETED" \
    --out-dir "$THESIS_DIR"
  echo "Wrote $THESIS_DIR/compare_to_ddae.csv"
}

run_gates() {
  echo "=== Integrity gates G-I1..G-I5 ==="
  "$PYTHON" scripts/validate_gates.py \
    --integrity \
    --completed "$FINAL_COMPLETED" \
    --compare "$THESIS_DIR/compare_to_ddae.json" \
    --logs-dir results/adadae_final/logs \
    --out "$THESIS_DIR/integrity_gates.json"
}

case "$MODE" in
  smoke) run_smoke ;;
  ddae) run_ddae ;;
  final) run_final ;;
  compare) run_compare ;;
  gates) run_gates ;;
  all)
    run_smoke
    run_ddae
    run_final
    run_compare
    run_gates
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: smoke|ddae|final|compare|gates|all"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
