#!/usr/bin/env bash
# Hard-tail + bleed ship path: select → freeze → invalidate → retrain → gates
# Integrity: val_loss selection only. Run on Vast (16gb).
#
# Usage:
#   bash scripts/run_hard_tail_ship_path.sh select 16gb
#   bash scripts/run_hard_tail_ship_path.sh freeze
#   bash scripts/run_hard_tail_ship_path.sh probe 16gb      # invalidate ship-probe + retrain those
#   bash scripts/run_hard_tail_ship_path.sh ship 16gb       # --all-semi + final 570 + gates
#   bash scripts/run_hard_tail_ship_path.sh all 16gb        # select→freeze→ship (long)
#   bash scripts/run_hard_tail_ship_path.sh paper-diag 16gb
#   bash scripts/run_hard_tail_ship_path.sh wire            # local: wire status JSONs only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

MODE="${1:-}"
HW="${2:-16gb}"

run_wire() {
  echo "=== Wire paper-diag + evidence freeze + dry-run (no GPU) ==="
  "$PYTHON" scripts/run_paper_protocol_diagnostic.py --wire-only
  "$PYTHON" scripts/train_only_recipe_select.py --apply-evidence-freeze
  "$PYTHON" scripts/train_only_recipe_select.py --dry-run | head -40
  echo "Catalog: results/adadae_per/thesis/weakness_catalog_20.json"
}

run_select() {
  echo "=== GPU val_loss select (--preset ship) ==="
  "$PYTHON" scripts/train_only_recipe_select.py --preset ship --seeds 111 222 333 --hardware "$HW"
}

run_freeze() {
  echo "=== Apply freeze (ship datasets: hard-12 + bleed-classical) ==="
  "$PYTHON" scripts/apply_hard_tail_freeze.py \
    --from results/adadae_per/thesis/phase1_hard_freeze.json \
    --datasets-preset ship
  "$PYTHON" scripts/phase0_revoke_audit.py || true
  "$PYTHON" scripts/train_only_recipe_select.py --dry-run | head -60
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
  bash scripts/run_adadae_per_protocol.sh gates
}

run_ship() {
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
  select) run_select ;;
  freeze) run_freeze ;;
  probe) run_probe ;;
  ship) run_ship ;;
  paper-diag) run_paper_diag ;;
  all)
    run_select
    run_freeze
    run_ship
    ;;
  *)
    echo "Usage: $0 {wire|select|freeze|probe|ship|paper-diag|all} [16gb]"
    exit 1
    ;;
esac

echo "Done mode=$MODE"
