#!/usr/bin/env bash
# AdaDDAE-6 protocol: smoke → hard → subset → LOO → full 570 → Table 6.
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
  echo "=== AdaDDAE-6 smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae6_final_smoke.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae6_final_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_smoke_hard() {
  echo "=== AdaDDAE-6 hard-subset smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae6_final_smoke_hard.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae6_final_smoke_hard.yaml \
    --datasets thyroid letter speech \
    --seeds 111 222 \
    --min-pr-auc 0.01 \
    "${HW_ARG[@]}"
}

run_subset() {
  echo "=== AdaDDAE-6 subset ladder ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae6_final.yaml \
    --steps adadae6_core helix_delta apex_calib kale_fuse full_adadae6 \
    --datasets cardio glass vertebral wine letter thyroid \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_loo() {
  echo "=== AdaDDAE-6 leave-one-out ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae6_final.yaml \
    --leave-one-out \
    --datasets cardio glass vertebral thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae6_final.yaml
  "$PYTHON" scripts/run_full_protocol.py --config configs/adadae6_final.yaml "${HW_ARG[@]}"
}

run_table6() {
  mkdir -p results/adadae6_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae6_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae6_final/thesis
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae6_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae6_final/thesis
  if [[ -f results/adadae5_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae6_final/metrics/completed.json \
      --baseline results/adadae5_final/metrics/completed.json \
      --out-dir results/adadae6_final/thesis/vs_table5
  fi
}

case "$MODE" in
  smoke) run_smoke ;;
  smoke_hard) run_smoke_hard ;;
  subset) run_subset ;;
  loo) run_loo ;;
  final) run_final ;;
  table6) run_table6 ;;
  all)
    run_smoke
    run_smoke_hard
    run_subset
    run_final
    run_table6
    ;;
  *) echo "modes: smoke|smoke_hard|subset|loo|final|table6|all"; exit 1 ;;
esac
echo "Done mode=$MODE"
