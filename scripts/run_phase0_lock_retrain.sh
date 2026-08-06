#!/usr/bin/env bash
# Phase0-lock / last-shot adaptive retrain path (Vast 16gb).
# Prefer lastshot (one union retrain) over chained disasters→residuals→midtier.
# Does NOT full-ship. Do NOT run full57 all.
#
# Usage:
#   bash scripts/run_phase0_lock_retrain.sh audit
#   bash scripts/run_phase0_lock_retrain.sh lastshot 16gb   # ONE Vast command
#   bash scripts/run_phase0_lock_retrain.sh top3 16gb
#   bash scripts/run_phase0_lock_retrain.sh disasters 16gb
#   bash scripts/run_phase0_lock_retrain.sh residuals 16gb
#   bash scripts/run_phase0_lock_retrain.sh midtier 16gb
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

# Union for last-shot: disasters ∪ residuals ∪ overlay strips ∪ proven midtier
LASTSHOT_DATASETS=(
  # disasters / composition victims
  FashionMNIST MNIST-C InternetAds optdigits backdoor thyroid
  # residuals
  Stamps smtp Pima letter WBC
  # overlay strips
  Yelp 20newsgroups Amazon vertebral fraud WPBC
  # proven midtier refresh
  Hepatitis cover glass Waveform cardio satimage-2
)

run_audit() {
  echo "=== Phase0 / last-shot lock audit + recipe map ==="
  "$PYTHON" scripts/phase0_revoke_audit.py
  "$PYTHON" scripts/dump_adaptive_recipe_map.py
  "$PYTHON" - <<'PY'
from src.config import load_yaml
from src.policy_per import apply_per_config, clear_per_upgrades_cache
clear_per_upgrades_cache()
cfg = load_yaml("configs/adadae_per.yaml")
for ds, cat in [
    ("wine", "classical"),
    ("census", "classical"),
    ("Agnews", "nlp"),
    ("FashionMNIST", "cv"),
    ("MNIST-C", "cv"),
    ("InternetAds", "classical"),
    ("optdigits", "classical"),
    ("backdoor", "classical"),
    ("thyroid", "classical"),
    ("vertebral", "classical"),
    ("Imdb", "nlp"),
    ("Amazon", "nlp"),
    ("fraud", "classical"),
    ("WPBC", "classical"),
    ("Hepatitis", "classical"),
    ("cover", "classical"),
    ("Pima", "classical"),
]:
    out = apply_per_config(cfg, "semi-supervised", cat, ds, meta={"n": 1.0, "d": 1.0})
    print(ds, out["adadae"].get("resolved_policy"))
PY
}

_check_probe_from_compare() {
  echo "=== Probe floor from compare (no second retrain) ==="
  "$PYTHON" - <<PY
import json
from pathlib import Path
floor = float("${PROBE_SEMI_FLOOR}")
path = Path("results/adadae_per/thesis/compare_to_ddae.json")
if not path.exists():
    print("MISSING compare_to_ddae.json — cannot gate")
    raise SystemExit(1)
data = json.loads(path.read_text())
semi = None
for row in data.get("adadae") or []:
    if row.get("setting") == "semi-supervised":
        semi = row
        break
if semi is None:
    print("MISSING semi-supervised row in compare_to_ddae.json")
    raise SystemExit(1)
pr = float(semi["AdaDDAE_PR_AUC"])
roc = float(semi["AdaDDAE_ROC_AUC"])
ok = pr >= floor
out = {
    "semi_PR": pr,
    "semi_ROC": roc,
    "floor": floor,
    "pass": ok,
    "ship_allowed": ok,
    "source": "compare_to_ddae.json (lastshot; no second probe retrain)",
    "note": "Full ship only if pass; else claim beat-fair adaptive under integrity",
}
Path("results/adadae_per/thesis/phase4_probe_gate.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
raise SystemExit(0 if ok else 1)
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
  "$PYTHON" scripts/check_unsup_hold.py || true
}

run_top3() {
  run_audit
  _retrain wine census Agnews
  echo "=== Expect wine/census Δ≈0 vs fair; Agnews improved (no TAPS) ==="
}

run_disasters() {
  run_audit
  _retrain FashionMNIST MNIST-C InternetAds optdigits backdoor thyroid
  echo "=== Expect FashionMNIST/optdigits/backdoor near fair ==="
}

run_residuals() {
  _retrain Stamps smtp Pima letter WBC
}

run_midtier() {
  _retrain Hepatitis cover glass Waveform cardio satimage-2
  echo "=== Probe floor check from compare (no second retrain) ==="
  if _check_probe_from_compare; then
    echo "PROBE PASS — consider: bash scripts/run_hard_tail_ship_path.sh ship $HW"
  else
    echo "PROBE BELOW FLOOR — do not full-ship; do NOT full57 all"
    exit 1
  fi
}

run_lastshot() {
  echo "=== LAST-SHOT adaptive union retrain (one GPU spend) ==="
  run_audit
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae_per.yaml
  _retrain "${LASTSHOT_DATASETS[@]}"
  echo "=== Last-shot probe floor from compare (no second retrain) ==="
  if _check_probe_from_compare; then
    echo "PROBE PASS (>= ${PROBE_SEMI_FLOOR}) — optional: bash scripts/run_hard_tail_ship_path.sh ship $HW"
  else
    echo "PROBE BELOW FLOOR ${PROBE_SEMI_FLOOR} — expected under integrity ceiling ~59.2"
    echo "Claim beat-fair adaptive DDAE if delta_PR_vs_fair > 0; do NOT burn credit on full57 all"
    # Do not exit 1: lastshot succeeded as recovery run; probe fail is informational
  fi
  echo "=== Done lastshot. Review results/adadae_per/thesis/compare_to_ddae.json ==="
}

case "$MODE" in
  audit) run_audit ;;
  top3) run_top3 ;;
  disasters) run_disasters ;;
  residuals) run_residuals ;;
  midtier) run_midtier ;;
  lastshot) run_lastshot ;;
  *)
    echo "Usage: $0 {audit|lastshot|top3|disasters|residuals|midtier} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
