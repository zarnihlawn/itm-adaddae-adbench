#!/usr/bin/env bash
# AdaDDAE-3 protocol: smoke → subset ladder → full 570 (GPU) → Table 3.
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
  echo "=== AdaDDAE-3 smoke ==="
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae3_final_smoke.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/smoke_final_integrity.py \
    --config configs/adadae3_final_smoke.yaml \
    --datasets cardio glass vertebral \
    --seeds 111 222 \
    "${HW_ARG[@]}"
}

run_subset() {
  echo "=== AdaDDAE-3 matched subset ladder ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae3_final.yaml \
    --steps adadae3_core helios_kairos orbis_plexus argos_aegis mirage_nexusv2 atlas_flux full_adadae3 \
    --datasets cardio glass vertebral wine letter thyroid Satimage ionosphere pendigits magic Waveform optdigits mammography shuttle pageblocks \
    --seeds 111 222 \
    "${HW_ARG[@]}" || true
}

run_loo() {
  echo "=== AdaDDAE-3 leave-one-out ==="
  "$PYTHON" scripts/ablations.py \
    --config configs/adadae3_final.yaml \
    --leave-one-out \
    --datasets cardio glass vertebral thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}" || true
}

run_tune() {
  echo "=== AdaDDAE-3 val-only tune (small grid) ==="
  "$PYTHON" scripts/tune_adadae3_valonly.py \
    --config configs/adadae3_final.yaml \
    --datasets cardio thyroid letter \
    --seeds 111 \
    "${HW_ARG[@]}"
}

run_final() {
  need_gpu
  "$PYTHON" scripts/assert_final_config.py --config configs/adadae3_final.yaml --allow-nonfinal-run-id
  "$PYTHON" scripts/run_full_protocol.py --config configs/adadae3_final.yaml "${HW_ARG[@]}"
}

run_table3() {
  mkdir -p results/adadae3_final/thesis
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae3_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae3_final/thesis
  "$PYTHON" scripts/stats_table1.py \
    --completed results/adadae3_final/metrics/completed.json \
    --baseline results/ddae_baseline_valstop/metrics/completed.json \
    --out-dir results/adadae3_final/thesis
  if [[ -f results/adadae_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae3_final/metrics/completed.json \
      --baseline results/adadae_final/metrics/completed.json \
      --out-dir results/adadae3_final/thesis/vs_table1
  fi
  if [[ -f results/adadae2_final/metrics/completed.json ]]; then
    "$PYTHON" scripts/stats_table1.py \
      --completed results/adadae3_final/metrics/completed.json \
      --baseline results/adadae2_final/metrics/completed.json \
      --out-dir results/adadae3_final/thesis/vs_table2
  fi
}

case "$MODE" in
  smoke) run_smoke ;;
  subset) run_subset ;;
  loo) run_loo ;;
  tune) run_tune ;;
  final) run_final ;;
  table3) run_table3 ;;
  all)
    run_smoke
    run_subset
    run_final
    run_table3
    ;;
  *) echo "modes: smoke|subset|loo|tune|final|table3|all"; exit 1 ;;
esac
echo "Done mode=$MODE"
