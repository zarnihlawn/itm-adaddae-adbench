#!/usr/bin/env bash
# AdaDDAE-5 protocol: smoke → subset → LOO → tune → full 570 → Table 5.
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

run_smoke() {
  echo "=== AdaDDAE-5 smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae5_final_smoke.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae5_final_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_smoke_hard() {
  echo "=== AdaDDAE-5 hard-subset smoke (thyroid/letter/speech) ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae5_final_smoke_hard.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae5_final_smoke_hard.yaml \
    --datasets thyroid letter speech \
    --seeds 111 222 \
    --min-pr-auc 0.01 \
    "${HW_ARG[@]}"
}

run_subset() {
  echo "=== AdaDDAE-5 subset ladder ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae5_final.yaml \
    --steps adadae5_core figaro_dsm mahala_dte lexicon_fuse full_adadae5 \
    --datasets cardio glass vertebral wine letter thyroid Satimage ionosphere pendigits magic Waveform optdigits mammography shuttle pageblocks \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_loo() {
  echo "=== AdaDDAE-5 leave-one-out ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae5_final.yaml \
    --leave-one-out \
    --datasets cardio glass vertebral thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}"
}

run_tune() {
  echo "=== AdaDDAE-5 val-only tune ==="
  "$PYTHON" scripts/tune_adadae5_valonly.py \
    --config configs/adadae5_final.yaml \
    --datasets cardio thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae5_final.yaml
  "$PYTHON" scripts/run_full_protocol.py --config configs/adadae5_final.yaml "${HW_ARG[@]}"
}

run_table5() {
  mkdir -p results/adadae5_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae5_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae5_final/thesis
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae5_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae5_final/thesis
  if [[ -f results/adadae_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae5_final/metrics/completed.json \
      --baseline results/adadae_final/metrics/completed.json \
      --out-dir results/adadae5_final/thesis/vs_table1
  fi
}

case "$MODE" in
  smoke) run_smoke ;;
  smoke_hard) run_smoke_hard ;;
  subset) run_subset ;;
  loo) run_loo ;;
  tune) run_tune ;;
  final) run_final ;;
  table5) run_table5 ;;
  all)
    run_smoke
    run_smoke_hard
    run_subset
    run_final
    run_table5
    ;;
  *) echo "modes: smoke|smoke_hard|subset|loo|tune|final|table5|all"; exit 1 ;;
esac
echo "Done mode=$MODE"
