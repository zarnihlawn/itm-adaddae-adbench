#!/usr/bin/env bash
# AdaDDAE-4 protocol: smoke → subset → LOO → tune → full 570 → Table 4 (+ regime report).
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then PYTHON=".venv/bin/python"; else PYTHON="${PYTHON:-python3}"; fi

MODE="${1:-smoke}"
HARDWARE="${2:-}"
HW_ARG=()
[[ -n "$HARDWARE" ]] && HW_ARG=(--hardware "$HARDWARE")

need_gpu() {
  if ! "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "ERROR: CUDA required for mode=$MODE"
    exit 2
  fi
}

run_audit() {
  echo "=== ADBench regime audit ==="
  "$PYTHON" scripts/adbench_regime_audit.py --out results/thesis/adbench_regimes.csv
}

run_smoke() {
  echo "=== AdaDDAE-4 smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae4_final_smoke.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae4_final_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_subset() {
  echo "=== AdaDDAE-4 subset ladder ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae4_final.yaml \
    --steps adadae4_core omni_gates full_adadae4 \
    --datasets cardio glass vertebral wine letter thyroid Satimage ionosphere pendigits magic Waveform optdigits mammography shuttle pageblocks \
    --seeds 111 222 \
    "${HW_ARG[@]}" || true
}

run_loo() {
  echo "=== AdaDDAE-4 leave-one-out ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae4_final.yaml \
    --leave-one-out \
    --datasets cardio glass vertebral thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}" || true
}

run_tune() {
  echo "=== AdaDDAE-4 val-only tune ==="
  "$PYTHON" scripts/tune_adadae4_valonly.py \
    --config configs/adadae4_final.yaml \
    --datasets cardio thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae4_final.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py --config configs/adadae4_final.yaml "${HW_ARG[@]}"
}

run_table4() {
  mkdir -p results/adadae4_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae4_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae4_final/thesis
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae4_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae4_final/thesis
  "$PYTHON" scripts/regime_eval_report.py \
    --completed results/adadae4_final/metrics/completed.json \
    --regimes results/thesis/adbench_regimes.csv \
    --out results/adadae4_final/thesis/regime_breakdown.json || true
  if [[ -f results/adadae_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae4_final/metrics/completed.json \
      --baseline results/adadae_final/metrics/completed.json \
      --out-dir results/adadae4_final/thesis/vs_table1
  fi
}

case "$MODE" in
  audit) run_audit ;;
  smoke) run_smoke ;;
  subset) run_subset ;;
  loo) run_loo ;;
  tune) run_tune ;;
  final) run_final ;;
  table4) run_table4 ;;
  all)
    run_audit
    run_smoke
    run_subset
    run_final
    run_table4
    ;;
  *) echo "modes: audit|smoke|subset|loo|tune|final|table4|all"; exit 1 ;;
esac
echo "Done mode=$MODE"
