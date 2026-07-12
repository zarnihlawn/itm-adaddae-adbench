#!/usr/bin/env bash
# Full pre-protocol smoke on a rented Vast instance: benchmark + ALOI + cardio.
# Usage: bash scripts/vast_smoke.sh [hardware_tier]
#   hardware_tier: 8gb | 12gb | 16gb | rtx5070ti (optional; auto-detect if omitted)
set -euo pipefail
cd "$(dirname "$0")/.."

HARDWARE="${1:-}"
HW_ARG=()
if [[ -n "$HARDWARE" ]]; then
  HW_ARG=(--hardware "$HARDWARE")
fi

echo "=== 1/4 Bootstrap (venv + cardio smoke) ==="
if [[ -n "$HARDWARE" ]]; then
  bash scripts/setup_vast.sh "$HARDWARE"
else
  bash scripts/setup_vast.sh
  HARDWARE="$(python scripts/detect_hardware.py 2>/dev/null | awk -F= '/^suggested_profile=/{print $2}' | sed 's/hardware_//;s/\.yaml$//')"
  if [[ -n "${HARDWARE:-}" ]]; then
    HW_ARG=(--hardware "$HARDWARE")
  fi
fi

source .venv/bin/activate

echo "=== 2/4 GPU benchmark (10 datasets, 30 epochs) ==="
python scripts/benchmark_gpu.py --config configs/default_gpu.yaml "${HW_ARG[@]}"

echo "=== 3/4 Hard dataset smoke: ALOI unsupervised ==="
python scripts/run_one.py \
  --config configs/default_gpu.yaml \
  --dataset ALOI \
  --setting unsupervised \
  --seed 111 \
  "${HW_ARG[@]}"

echo "=== 4/4 Hard dataset smoke: cardio semi-supervised ==="
python scripts/run_one.py \
  --config configs/default_gpu.yaml \
  --dataset cardio \
  --setting semi-supervised \
  --seed 111 \
  "${HW_ARG[@]}"

echo ""
echo "Smoke passed. Start full 570-job protocol (resumable):"
echo "  python scripts/run_full_protocol.py --config configs/default_gpu.yaml ${HW_ARG[*]}"
echo ""
echo "Before terminating the instance, sync results back:"
echo "  bash scripts/sync_results_from_vast.sh <ssh-host>"
