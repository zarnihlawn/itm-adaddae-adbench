#!/usr/bin/env bash
# Hard-tail + bleed ship path: select → freeze → invalidate → retrain → gates
# Integrity: val_loss (+ ε/complexity/synth-val) selection — never test-PR. Vast 16gb.
#
# Usage:
#   bash scripts/run_hard_tail_ship_path.sh phase0 16gb   # emergency revoke retrain
#   bash scripts/run_hard_tail_ship_path.sh select 16gb
#   bash scripts/run_hard_tail_ship_path.sh freeze
#   bash scripts/run_hard_tail_ship_path.sh probe 16gb      # ship-probe + gate; abort ship if weak
#   bash scripts/run_hard_tail_ship_path.sh ship 16gb       # --all-semi + final 570 + gates
#   bash scripts/run_hard_tail_ship_path.sh all 16gb        # select→freeze→probe (not auto-ship)
#   bash scripts/run_hard_tail_ship_path.sh paper-diag 16gb
#   bash scripts/run_hard_tail_ship_path.sh wire            # local: wire status JSONs only
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
# Phase 4: only full-ship if probe semi PR clears this floor
PROBE_SEMI_FLOOR="${PROBE_SEMI_FLOOR:-60.5}"

run_wire() {
  echo "=== Wire paper-diag + evidence freeze + dry-run (no GPU) ==="
  "$PYTHON" scripts/run_paper_protocol_diagnostic.py --wire-only
  "$PYTHON" scripts/train_only_recipe_select.py --apply-evidence-freeze
  "$PYTHON" scripts/train_only_recipe_select.py --dry-run | head -40
  "$PYTHON" scripts/verify_scoring_parity.py || true
  echo "Catalog: results/adadae_per/thesis/weakness_catalog_20.json"
}

run_phase0() {
  echo "=== Phase 0 emergency revoke retrain (wine/census/smtp/Stamps/celeba/WBC) ==="
  "$PYTHON" scripts/invalidate_per_semi_jobs.py \
    --datasets wine census smtp Stamps celeba WBC
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_per.yaml \
    --hardware "$HW" \
    --datasets wine census smtp Stamps celeba WBC
  bash scripts/run_adadae_per_protocol.sh compare
  bash scripts/run_adadae_per_protocol.sh gates || true
  echo "Wrote/see results/adadae_per/thesis/phase0_emergency_revoke.json"
}

run_select() {
  echo "=== GPU val_loss+synth select (--preset ship, ε-ball+complexity) ==="
  "$PYTHON" scripts/train_only_recipe_select.py \
    --preset ship --seeds 111 222 333 --hardware "$HW" \
    --eps-rel 0.05 --rdt-veto-margin 0.05
}

run_freeze() {
  echo "=== Apply freeze (ship datasets: hard-12 + bleed-classical) ==="
  "$PYTHON" scripts/apply_hard_tail_freeze.py \
    --from results/adadae_per/thesis/phase1_hard_freeze.json \
    --datasets-preset ship
  "$PYTHON" scripts/phase0_revoke_audit.py || true
  "$PYTHON" scripts/train_only_recipe_select.py --dry-run | head -60
}

_check_probe_floor() {
  "$PYTHON" - <<PY
import json
from pathlib import Path
p = Path("results/adadae_per/thesis/compare_to_ddae.json")
floor = float("${PROBE_SEMI_FLOOR}")
if not p.exists():
    print("MISSING compare_to_ddae.json — cannot gate ship")
    raise SystemExit(2)
data = json.loads(p.read_text())
rows = {r["setting"]: r for r in data.get("adadae", [])}
semi = rows.get("semi-supervised") or {}
pr = float(semi.get("AdaDDAE_PR_AUC", 0))
roc = float(semi.get("AdaDDAE_ROC_AUC", 0))
print(f"probe semi PR={pr:.4f} ROC={roc:.4f} floor={floor}")
ok = pr >= floor
Path("results/adadae_per/thesis/phase4_probe_gate.json").write_text(
    json.dumps({
        "semi_PR": pr,
        "semi_ROC": roc,
        "floor": floor,
        "pass": ok,
        "ship_allowed": ok,
        "note": "Full ship only if pass; else iterate selector/method lifts",
    }, indent=2)
)
raise SystemExit(0 if ok else 1)
PY
}

run_probe() {
  echo "=== Invalidate ship-probe + retrain those datasets ==="
  "$PYTHON" scripts/invalidate_per_semi_jobs.py --ship-probe --dry-run
  "$PYTHON" scripts/invalidate_per_semi_jobs.py --ship-probe
  "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_per.yaml \
    --hardware "$HW" \
    --datasets speech ALOI celeba SVHN CIFAR10 Wilt \
               Imdb Amazon Yelp Agnews 20newsgroups census \
               smtp satimage-2 Pima Stamps letter wine
  bash scripts/run_adadae_per_protocol.sh compare
  bash scripts/run_adadae_per_protocol.sh gates || true
  echo "=== Probe floor check (semi PR >= ${PROBE_SEMI_FLOOR}) ==="
  if _check_probe_floor; then
    echo "PROBE PASS — safe to consider: bash $0 ship $HW"
  else
    echo "PROBE BELOW FLOOR — DO NOT full-ship. Iterate Phase 1–3."
    exit 1
  fi
}

run_ship() {
  echo "=== Pre-ship probe floor check ==="
  if ! _check_probe_floor; then
    echo "Refusing ship: probe semi PR < ${PROBE_SEMI_FLOOR}. Run probe first / iterate."
    exit 1
  fi
  echo "=== Full ship: invalidate all semi + final 570 + gates ==="
  "$PYTHON" scripts/invalidate_per_semi_jobs.py --all-semi
  bash scripts/run_adadae_per_protocol.sh final "$HW"
  bash scripts/run_adadae_per_protocol.sh compare
  bash scripts/run_adadae_per_protocol.sh gates
  "$PYTHON" scripts/check_unsup_hold.py || true
  "$PYTHON" scripts/audit_ap_pr_consistency.py \
    --completed results/adadae_per/metrics/completed.json || true
}

run_paper_diag() {
  echo "=== Paper-protocol diagnostic (appendix, not ship) ==="
  "$PYTHON" scripts/run_paper_protocol_diagnostic.py \
    --seeds 111 222 --hardware "$HW"
}

case "$MODE" in
  wire) run_wire ;;
  phase0) run_phase0 ;;
  select) run_select ;;
  freeze) run_freeze ;;
  probe) run_probe ;;
  ship) run_ship ;;
  paper-diag) run_paper_diag ;;
  all)
    run_select
    run_freeze
    run_probe
    ;;
  *)
    echo "Usage: $0 {wire|phase0|select|freeze|probe|ship|paper-diag|all} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
