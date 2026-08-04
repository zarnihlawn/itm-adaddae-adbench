#!/usr/bin/env bash
# AdaDDAE champion protocol — ONE primary full-run path (replaces Tables 1–6).
#
# Usage:
#   bash scripts/run_adadae_champion_protocol.sh smoke [hardware]
#   bash scripts/run_adadae_champion_protocol.sh hard_subset [hardware]
#   bash scripts/run_adadae_champion_protocol.sh ddae [hardware]       # fair DDAE 570 if missing
#   bash scripts/run_adadae_champion_protocol.sh final [hardware]      # champion 570
#   bash scripts/run_adadae_champion_protocol.sh compare
#   bash scripts/run_adadae_champion_protocol.sh gates                 # integrity + paper-both
#   bash scripts/run_adadae_champion_protocol.sh all [hardware]
#
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

CHAMPION_COMPLETED="results/adadae_champion/metrics/completed.json"
DDAE_COMPLETED="results/ddae_baseline_valstop/metrics/completed.json"
THESIS_DIR="results/adadae_champion/thesis"

# Hard classical + known tails for pre-570 triage (both settings, 2 seeds)
HARD_DATASETS=(
  cardio glass vertebral wine letter thyroid speech ALOI
  pendigits musk shuttle mammography
)

need_gpu() {
  if ! "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: CUDA required for mode '$MODE' (full / hard_subset). Rent Vast and re-run."
    echo "  Example: bash scripts/run_adadae_champion_protocol.sh all 16gb"
    exit 2
  fi
}

run_smoke() {
  echo "=== Champion smoke integrity ==="
  "$PYTHON" scripts/assert_final_config.py \
    --config configs/adadae_champion_smoke.yaml \
    --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae_champion_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_hard_subset() {
  need_gpu
  echo "=== Champion hard subset (${#HARD_DATASETS[@]} datasets × 2 settings × seeds 111,222) ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae_champion.yaml
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_champion.yaml \
    --datasets "${HARD_DATASETS[@]}" \
    --seeds 111 222 \
    "${HW_ARG[@]}"
  mkdir -p "$THESIS_DIR"
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed "$CHAMPION_COMPLETED" \
    --baseline "$DDAE_COMPLETED" \
    --out-dir "$THESIS_DIR/hard_subset"
  echo "Hard-subset compare → $THESIS_DIR/hard_subset (paper-both not required yet)"
}

run_ddae() {
  need_gpu
  if [[ -f "$DDAE_COMPLETED" ]]; then
    n="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("results/ddae_baseline_valstop/metrics/completed.json")
print(len(json.loads(p.read_text()).get("completed", {})))
PY
)"
    if [[ "$n" -ge 570 ]]; then
      echo "=== Fair DDAE already complete ($n/570) — skip ==="
      return 0
    fi
    echo "=== Fair DDAE resume ($n/570) ==="
  else
    echo "=== Fair DDAE baseline 570 (val-only early stop) ==="
  fi
  "$PYTHON" scripts/assert_final_config.py \
    --config configs/baselines_ddae_valstop.yaml \
    --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/baselines_ddae_valstop.yaml \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  echo "=== AdaDDAE champion 570 (paradigm recipe) ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae_champion.yaml
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_champion.yaml \
    "${HW_ARG[@]}"
}

run_compare() {
  echo "=== Champion compare vs paper + fair DDAE ==="
  mkdir -p "$THESIS_DIR"
  if [[ ! -f "$CHAMPION_COMPLETED" ]]; then
    echo "Missing $CHAMPION_COMPLETED"
    exit 1
  fi
  if [[ ! -f "$DDAE_COMPLETED" ]]; then
    echo "Missing $DDAE_COMPLETED — run ddae track first."
    exit 1
  fi
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed "$CHAMPION_COMPLETED" \
    --baseline "$DDAE_COMPLETED" \
    --out-dir "$THESIS_DIR"
  "$PYTHON" scripts/stats_table1.py \
    --completed "$CHAMPION_COMPLETED" \
    --baseline "$DDAE_COMPLETED" \
    --out-dir "$THESIS_DIR" || true
  echo "Wrote $THESIS_DIR/compare_to_ddae.csv"
}

run_gates() {
  echo "=== Integrity + paper-both ship gate ==="
  "$PYTHON" scripts/validate_gates.py \
    --integrity \
    --paper-both \
    --completed "$CHAMPION_COMPLETED" \
    --compare "$THESIS_DIR/compare_to_ddae.json" \
    --logs-dir results/adadae_champion/logs \
    --out "$THESIS_DIR/integrity_gates.json"
}

case "$MODE" in
  smoke) run_smoke ;;
  hard_subset) run_hard_subset ;;
  ddae) run_ddae ;;
  final) run_final ;;
  compare) run_compare ;;
  gates) run_gates ;;
  all)
    run_smoke
    run_ddae
    run_hard_subset
    run_final
    run_compare
    run_gates
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: smoke|hard_subset|ddae|final|compare|gates|all"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
