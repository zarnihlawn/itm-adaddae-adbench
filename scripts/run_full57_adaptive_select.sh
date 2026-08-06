#!/usr/bin/env bash
# Full-57 integrity-safe adaptive select (fair vs paper / fair DDAE).
#
# AdaDDAE = Adaptive Diffusion: per-dataset recipes chosen on val_loss only
# (ε-ball + complexity + synth-val). NEVER test-PR.
#
# Prerequisites (required before GPU select):
#   bash scripts/run_phase0_lock_retrain.sh top3 16gb
#   (wine/census/Agnews must already resolve to PHASE0-locked policies)
#
# Usage:
#   bash scripts/run_full57_adaptive_select.sh plan          # local: tier plan only
#   bash scripts/run_full57_adaptive_select.sh check-top3    # gate: Phase0 metrics OK?
#   bash scripts/run_full57_adaptive_select.sh select 16gb   # Vast GPU select
#   bash scripts/run_full57_adaptive_select.sh freeze        # apply winners → YAML
#   bash scripts/run_full57_adaptive_select.sh final 16gb    # invalidate+570+gates
#   bash scripts/run_full57_adaptive_select.sh all 16gb      # select→freeze→final→probe
#   bash scripts/run_full57_adaptive_select.sh probe 16gb
#   bash scripts/run_full57_adaptive_select.sh ship 16gb     # only if probe PASS
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

check_top3() {
  echo "=== Require Phase0 top3 retrain before full-57 GPU spend ==="
  "$PYTHON" - <<'PY'
import json
import sys
from pathlib import Path

root = Path(".")
metrics = root / "results/adadae_per/metrics"
need = {
    "wine": "baseline_ddae",
    "census": "baseline_ddae",
    "Agnews": "baseline_ddae",  # no taps
}
errors = []
for ds, needle in need.items():
    p = metrics / f"{ds}__semi-supervised__111.json"
    if not p.exists():
        errors.append(f"MISSING {p}")
        continue
    data = json.loads(p.read_text())
    pol = str(data.get("resolved_policy") or "")
    if needle not in pol:
        errors.append(f"{ds}: resolved_policy={pol!r} (need {needle})")
    if ds == "wine" and "champion" in pol:
        errors.append(f"{ds}: still champion_semi — run top3 retrain")
    if ds == "census" and "rdt" in pol:
        errors.append(f"{ds}: still RDT — run top3 retrain")
    if ds == "Agnews" and "taps" in pol:
        errors.append(f"{ds}: TAPS still on — run top3 retrain")

out = {
    "pass": not errors,
    "errors": errors,
    "fix": "bash scripts/run_phase0_lock_retrain.sh top3 16gb",
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
}

run_select() {
  if [[ "$SKIP_TOP3_CHECK" != "1" ]]; then
    if ! check_top3; then
      echo "REFUSING full-57 select: Phase0 top3 metrics not locked yet."
      echo "Run: bash scripts/run_phase0_lock_retrain.sh top3 16gb"
      echo "Or set SKIP_TOP3_CHECK=1 to override."
      exit 2
    fi
  else
    echo "WARN: SKIP_TOP3_CHECK=1 — proceeding without top3 gate"
  fi
  echo "=== GPU all-semi select (val_loss + ε + complexity + synth-val) ==="
  "$PYTHON" scripts/train_only_recipe_select.py \
    --preset all-semi --seeds 111 222 333 \
    --hardware "$HW" --eps-rel 0.05 --rdt-veto-margin 0.05
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
    run_select
    run_freeze
    run_final
    echo "=== Post-final probe floor ==="
    if bash scripts/run_hard_tail_ship_path.sh probe "$HW"; then
      echo "PROBE PASS — consider: bash $0 ship $HW"
    else
      echo "PROBE BELOW FLOOR — do not ship; iterate search/method lifts"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 {plan|check-top3|select|freeze|final|probe|ship|all} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
