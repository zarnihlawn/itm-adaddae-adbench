#!/usr/bin/env bash
# AdaDDAE-PER protocol — ONE full-run path (frozen v2→v5.1 hybrid rules).
#
# Usage:
#   bash scripts/run_adadae_per_protocol.sh smoke [hardware]
#   bash scripts/run_adadae_per_protocol.sh final [hardware]   # full 570
#   bash scripts/run_adadae_per_protocol.sh ddae [hardware]    # fair DDAE if missing
#   bash scripts/run_adadae_per_protocol.sh compare
#   bash scripts/run_adadae_per_protocol.sh gates              # paper-both ship gate
#   bash scripts/run_adadae_per_protocol.sh dump_routing       # print policy map
#   bash scripts/run_adadae_per_protocol.sh all [hardware]
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

PER_COMPLETED="results/adadae_per/metrics/completed.json"
DDAE_COMPLETED="results/ddae_baseline_valstop/metrics/completed.json"
THESIS_DIR="results/adadae_per/thesis"

need_gpu() {
  if ! "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: CUDA required for mode '$MODE'. Rent Vast and re-run."
    echo "  Example: bash scripts/run_adadae_per_protocol.sh all 16gb"
    exit 2
  fi
}

run_dump_routing() {
  echo "=== AdaDDAE-PER routing map (57 × 2) ==="
  "$PYTHON" scripts/dump_adadae_per_routing.py
}

run_smoke() {
  echo "=== AdaDDAE-PER smoke integrity ==="
  "$PYTHON" scripts/assert_final_config.py \
    --config configs/adadae_per_smoke.yaml \
    --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae_per_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_ddae() {
  need_gpu
  if [[ -f "$DDAE_COMPLETED" ]]; then
    N=$("$PYTHON" -c "import json; print(len(json.load(open('$DDAE_COMPLETED'))['completed']))")
    if [[ "$N" -ge 570 ]]; then
      echo "=== Fair DDAE already complete ($N/570) — skip ==="
      return 0
    fi
  fi
  echo "=== Fair DDAE 570 (val_loss early stop) ==="
  "$PYTHON" scripts/assert_final_config.py \
    --config configs/baselines_ddae_valstop.yaml \
    --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/baselines_ddae_valstop.yaml \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  echo "=== AdaDDAE-PER full 570 (single model, policy=per) ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae_per.yaml
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_per.yaml \
    "${HW_ARG[@]}"
}

run_compare() {
  echo "=== PER compare vs paper + fair DDAE ==="
  mkdir -p "$THESIS_DIR"
  if [[ ! -f "$PER_COMPLETED" ]]; then
    echo "Missing $PER_COMPLETED"
    exit 1
  fi
  if [[ ! -f "$DDAE_COMPLETED" ]]; then
    echo "Missing $DDAE_COMPLETED — run ddae track first (or skip fair deltas)."
  fi
  ARGS=(--completed "$PER_COMPLETED" --out-dir "$THESIS_DIR")
  if [[ -f "$DDAE_COMPLETED" ]]; then
    ARGS+=(--baseline "$DDAE_COMPLETED")
  fi
  "$PYTHON" scripts/compare_to_ddae.py "${ARGS[@]}"
  if [[ -f "$DDAE_COMPLETED" ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed "$PER_COMPLETED" \
      --baseline "$DDAE_COMPLETED" \
      --out-dir "$THESIS_DIR" || true
  fi
  echo "Wrote $THESIS_DIR/compare_to_ddae.csv"
}

run_gates() {
  echo "=== Paper-both ship gate (AdaDDAE-PER) ==="
  if [[ ! -f "$THESIS_DIR/compare_to_ddae.json" ]]; then
    run_compare
  fi
  # Completeness + paper-both. Routing is intentional for PER (skip G-I2 via
  # paper-both on compare only; still require 570 jobs).
  "$PYTHON" - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.validate_gates import check_paper_both, check_gi1_complete_570

completed_path = Path("results/adadae_per/metrics/completed.json")
compare = Path("results/adadae_per/thesis/compare_to_ddae.json")
out = Path("results/adadae_per/thesis/integrity_gates.json")

completed = json.loads(completed_path.read_text())["completed"]
gates = {
    "G-I1_complete_570": check_gi1_complete_570(completed),
    "G_paper_both": check_paper_both(compare),
    "model": "adadae_per",
    "note": "Routing/MCE/SMC/GATE are intentional (frozen v5.1 rules); G-I2 skipped.",
}
gates["all_pass"] = bool(gates["G-I1_complete_570"]["pass"] and gates["G_paper_both"]["pass"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(gates, indent=2), encoding="utf-8")
print(f"G-I1_complete_570: {'PASS' if gates['G-I1_complete_570']['pass'] else 'FAIL'}")
print(f"G_paper_both: {'PASS' if gates['G_paper_both']['pass'] else 'FAIL'}")
print(f"ALL PASS: {gates['all_pass']}")
print(f"Wrote {out}")
sys.exit(0 if gates["all_pass"] else 1)
PY
}

case "$MODE" in
  dump_routing) run_dump_routing ;;
  smoke) run_smoke ;;
  ddae) run_ddae ;;
  final) run_final ;;
  compare) run_compare ;;
  gates) run_gates ;;
  all)
    run_dump_routing
    run_smoke
    run_ddae
    run_final
    run_compare
    run_gates
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Use: dump_routing|smoke|ddae|final|compare|gates|all"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
