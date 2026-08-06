#!/usr/bin/env bash
# Phase0-lock retrain path (Vast 16gb) — after wine/census/Agnews config locks.
# Does NOT full-ship. Stop if compare still < PROBE_SEMI_FLOOR.
#
# Usage:
#   bash scripts/run_phase0_lock_retrain.sh top3 16gb     # wine census Agnews
#   bash scripts/run_phase0_lock_retrain.sh residuals 16gb # Stamps smtp Pima letter WBC
#   bash scripts/run_phase0_lock_retrain.sh midtier 16gb   # cal_fuse winners + mid-tier
#   bash scripts/run_phase0_lock_retrain.sh audit          # local: phase0 audit + dry-run
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
fi

MODE="${1:-}"
HW="${2:-16gb}"
PROBE_SEMI_FLOOR="${PROBE_SEMI_FLOOR:-60.5}"

run_audit() {
  echo "=== Phase0 lock audit + resolve dry-run ==="
  "$PYTHON" scripts/phase0_revoke_audit.py
  "$PYTHON" scripts/train_only_recipe_select.py --dry-run | head -50
  "$PYTHON" - <<'PY'
from src.config import load_yaml
from src.policy_per import apply_per_config, clear_per_upgrades_cache
clear_per_upgrades_cache()
cfg = load_yaml("configs/adadae_per.yaml")
for ds, cat in [
    ("wine", "classical"),
    ("census", "classical"),
    ("Agnews", "nlp"),
    ("Pima", "classical"),
    ("letter", "classical"),
    ("backdoor", "classical"),
    ("Hepatitis", "classical"),
]:
    out = apply_per_config(cfg, "semi-supervised", cat, ds, meta={"n": 1.0, "d": 1.0})
    print(ds, out["adadae"].get("resolved_policy"))
PY
}

_retrain() {
  local -a ds=("$@")
  echo "=== Invalidate + retrain: ${ds[*]} ==="
  "$PYTHON" scripts/invalidate_per_semi_jobs.py --datasets "${ds[@]}"
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_per.yaml \
    --hardware "$HW" \
    --datasets "${ds[@]}"
  bash scripts/run_adadae_per_protocol.sh compare
  bash scripts/run_adadae_per_protocol.sh gates || true
  "$PYTHON" scripts/phase0_revoke_audit.py || true
}

run_top3() {
  run_audit
  _retrain wine census Agnews
  echo "=== Expect wine/census Δ≈0 vs fair; Agnews improved (no TAPS) ==="
}

run_residuals() {
  _retrain Stamps smtp Pima letter WBC
}

run_midtier() {
  # Proven cal_fuse expand + mid-tier campaign (never wine/census RDT)
  _retrain backdoor Hepatitis cover annthyroid campaign yeast MVTec-AD cardio
  echo "=== Probe floor check (semi PR >= ${PROBE_SEMI_FLOOR}) ==="
  if bash scripts/run_hard_tail_ship_path.sh probe "$HW"; then
    echo "PROBE PASS — consider: bash scripts/run_hard_tail_ship_path.sh ship $HW"
  else
    echo "PROBE BELOW FLOOR — iterate mid-tier; do not full-ship"
    exit 1
  fi
}

case "$MODE" in
  audit) run_audit ;;
  top3) run_top3 ;;
  residuals) run_residuals ;;
  midtier) run_midtier ;;
  *)
    echo "Usage: $0 {audit|top3|residuals|midtier} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
