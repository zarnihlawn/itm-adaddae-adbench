#!/usr/bin/env bash
# AdaDDAE-2 protocol: smoke → subset ladder → full 570 (GPU) → Table 2.
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
  echo "=== AdaDDAE-2 smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae2_final_smoke.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae2_final_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_subset() {
  echo "=== AdaDDAE-2 matched subset ladder (15 datasets) ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae2_final.yaml \
    --steps adadae2_core chronos geode calix nexus aether full_adadae2 \
    --datasets cardio glass vertebral wine letter thyroid Satimage ionosphere pendigits magic Waveform optdigits mammography shuttle pageblocks \
    --seeds 111 222 \
    "${HW_ARG[@]}" || true
}

run_final() {
  need_gpu
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae2_final.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py --config configs/adadae2_final.yaml "${HW_ARG[@]}"
}

run_table2() {
  mkdir -p results/adadae2_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae2_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae2_final/thesis
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae2_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae2_final/thesis
  # Also vs Table 1 if present
  if [[ -f results/adadae_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae2_final/metrics/completed.json \
      --baseline results/adadae_final/metrics/completed.json \
      --out-dir results/adadae2_final/thesis/vs_table1
  fi
}

case "$MODE" in
  smoke) run_smoke ;;
  subset) run_subset ;;
  final) run_final ;;
  table2) run_table2 ;;
  all)
    run_smoke
    run_subset
    run_final
    run_table2
    ;;
  *) echo "modes: smoke|subset|final|table2|all"; exit 1 ;;
esac
echo "Done mode=$MODE"
