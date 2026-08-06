#!/usr/bin/env bash
# Full-57 integrity-safe adaptive select (fair vs paper / fair DDAE).
#
# AdaDDAE = Adaptive Diffusion: per-dataset recipes chosen on val_loss only
# (ε-ball + complexity + synth-val + composed upgrades). NEVER test-PR.
#
# WARNING (2026-08-06): full57 all regressed semi 59.09 → 57.52 (RDT disasters +
# taps/cal_fuse composition). Prefer:
#   bash scripts/run_phase0_lock_retrain.sh disasters 16gb
#   bash scripts/run_phase0_lock_retrain.sh residuals 16gb
#   bash scripts/run_phase0_lock_retrain.sh midtier 16gb
# Re-running select/all requires CONFIRM_FULL57_ALL=1 after selector fixes land.
#
# Prerequisites:
#   bash scripts/run_phase0_lock_retrain.sh top3 16gb
#   bash scripts/run_phase0_lock_retrain.sh disasters 16gb
#
# Usage:
#   bash scripts/run_full57_adaptive_select.sh plan
#   bash scripts/run_full57_adaptive_select.sh check-top3
#   CONFIRM_FULL57_ALL=1 bash scripts/run_full57_adaptive_select.sh select 16gb
#   bash scripts/run_full57_adaptive_select.sh freeze
#   bash scripts/run_full57_adaptive_select.sh final 16gb
#   CONFIRM_FULL57_ALL=1 bash scripts/run_full57_adaptive_select.sh all 16gb
#   bash scripts/run_full57_adaptive_select.sh probe 16gb
#   bash scripts/run_full57_adaptive_select.sh ship 16gb
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
SKIP_TOP3_CHECK="${SKIP_TOP3_CHECK:-0}"
CONFIRM_FULL57_ALL="${CONFIRM_FULL57_ALL:-0}"

require_confirm_full57() {
  if [[ "$CONFIRM_FULL57_ALL" != "1" ]]; then
    echo "REFUSING full-57 ${MODE}: prior all-run crashed semi to 57.52."
    echo "Prefer: bash scripts/run_phase0_lock_retrain.sh disasters|residuals|midtier 16gb"
    echo "Only after composed-select + RDT ratio veto are verified, re-run with:"
    echo "  CONFIRM_FULL57_ALL=1 bash $0 ${MODE} $HW"
    exit 3
  fi
  echo "WARN: CONFIRM_FULL57_ALL=1 — proceeding with full-57 ${MODE}"
}

check_top3() {
  echo "=== Require Phase0 top3 + disaster locks before full-57 GPU spend ==="
  "$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path

root = Path(".")
metrics = root / "results/adadae_per/metrics"
need = {
    "wine": "baseline_ddae",
    "census": "baseline_ddae",
    "Agnews": "baseline_ddae",
    "FashionMNIST": "baseline_ddae",
    "MNIST-C": "baseline_ddae",
    "InternetAds": "baseline_ddae",
    "optdigits": "baseline_ddae",
    "backdoor": "baseline_ddae",
}
errors = []
for ds, needle in need.items():
    p = metrics / f"{ds}__semi-supervised__111.json"
    if not p.exists():
        errors.append(f"MISSING {p} — run disasters retrain")
        continue
    data = json.loads(p.read_text())
    pol = str(data.get("resolved_policy") or "")
    if needle not in pol:
        errors.append(f"{ds}: resolved_policy={pol!r} (need {needle})")
    if "rdt" in pol and ds != "speech":
        errors.append(f"{ds}: still RDT — run disasters retrain")
    if ds == "wine" and "champion" in pol:
        errors.append(f"{ds}: still champion_semi — run top3 retrain")
    if ds == "Agnews" and "taps" in pol:
        errors.append(f"{ds}: TAPS still on — run top3 retrain")
    if ds in ("backdoor", "optdigits", "thyroid") and "taps" in pol:
        errors.append(f"{ds}: TAPS still on — strip upgrades + disasters retrain")
    if ds in ("backdoor", "optdigits") and "cal_fuse" in pol:
        errors.append(f"{ds}: cal_fuse still on — strip upgrades + disasters retrain")

out = {
    "pass": not errors,
    "errors": errors,
    "fix": "bash scripts/run_phase0_lock_retrain.sh top3 16gb && bash scripts/run_phase0_lock_retrain.sh disasters 16gb",
    "override": "SKIP_TOP3_CHECK=1 to bypass (not recommended)",
}
Path("results/adadae_per/thesis/full57_top3_gate.json").write_text(
    json.dumps(out, indent=2)
)
print(json.dumps(out, indent=2))
sys.exit(0 if out["pass"] else 1)
PY
}

run_plan() {
  echo "=== Full-57 tier plan (no GPU) ==="
  "$PYTHON" scripts/train_only_recipe_select.py --tier-plan-only
  "$PYTHON" scripts/phase0_revoke_audit.py || true
  echo "See results/adadae_per/thesis/phase1_all_semi_tier_plan.json"
  echo "NOTE: prefer disasters→residuals→midtier over another full57 all"
}

run_select() {
  require_confirm_full57
  if [[ "$SKIP_TOP3_CHECK" != "1" ]]; then
    if ! check_top3; then
      echo "REFUSING full-57 select: Phase0 top3/disaster metrics not locked yet."
      echo "Run: bash scripts/run_phase0_lock_retrain.sh disasters 16gb"
      echo "Or set SKIP_TOP3_CHECK=1 to override."
      exit 2
    fi
  else
    echo "WARN: SKIP_TOP3_CHECK=1 — proceeding without top3/disaster gate"
  fi
  echo "=== GPU all-semi select (composed upgrades + val_loss ratio veto) ==="
  "$PYTHON" scripts/train_only_recipe_select.py \
    --preset all-semi --seeds 111 222 333 \
    --hardware "$HW" --eps-rel 0.05 --rdt-veto-margin 0.05 \
    --max-val-loss-ratio 2.5
}

run_freeze() {
  echo "=== Freeze all-semi winners into PER YAML ==="
  "$PYTHON" scripts/apply_hard_tail_freeze.py \
    --from results/adadae_per/thesis/phase1_hard_freeze.json \
    --datasets-preset all-semi
  "$PYTHON" scripts/phase0_revoke_audit.py
}

run_final() {
  echo "=== Invalidate all semi + final 570 + compare/gates ==="
  "$PYTHON" scripts/invalidate_per_semi_jobs.py --all-semi
  bash scripts/run_adadae_per_protocol.sh final "$HW"
  bash scripts/run_adadae_per_protocol.sh compare
  bash scripts/run_adadae_per_protocol.sh gates || true
  "$PYTHON" scripts/check_unsup_hold.py || true
}

run_probe() {
  echo "=== Probe floor (semi PR >= ${PROBE_SEMI_FLOOR}) ==="
  bash scripts/run_hard_tail_ship_path.sh probe "$HW"
}

run_ship() {
  bash scripts/run_hard_tail_ship_path.sh ship "$HW"
}

case "$MODE" in
  plan) run_plan ;;
  check-top3) check_top3 ;;
  select) run_select ;;
  freeze) run_freeze ;;
  final) run_final ;;
  probe) run_probe ;;
  ship) run_ship ;;
  all)
    require_confirm_full57
    run_select
    run_freeze
    run_final
    echo "=== Post-final probe floor ==="
    if bash scripts/run_hard_tail_ship_path.sh probe "$HW"; then
      echo "PROBE PASS — consider: bash $0 ship $HW"
    else
      echo "PROBE BELOW FLOOR — do not ship; iterate residuals/midtier (not another blind all)"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {plan|check-top3|select|freeze|final|probe|ship|all} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
