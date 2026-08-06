#!/usr/bin/env bash
# Run AXION probes on Vast GPU.
# Usage:
#   bash scripts/vast_probe.sh smoke
#   bash scripts/vast_probe.sh classical
#   bash scripts/vast_probe.sh g2
#   bash scripts/vast_probe.sh full57   # later phase — all 57 × 2 × 5
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

MODE="${1:-smoke}"
CFG="${2:-configs/default.yaml}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required on Vast for this script (or edit to allow CPU)"
print("GPU:", torch.cuda.get_device_name(0))
PY

case "$MODE" in
  smoke)
    python scripts/run_probe.py --config "$CFG" --smoke --model axion --loop-log --run-id axion_smoke
    ;;
  classical)
    # 8 classical × 2 settings × seeds 111,222,333 (design loop)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_classical \
      --datasets breastw cardio glass thyroid satimage-2 cover fraud backdoor \
      --seeds 111 222 333 \
      --max-train-samples 20000
    ;;
  g2)
    # Full Phase-2 G2 probe (12 datasets × 2 × 3 seeds). CV/NLP use all variants.
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g2 \
      --all-probe --max-train-samples 20000 --max-variants 0
    ;;
  g2-fast)
    # Faster G2: 1 seed, 3 CV/NLP variants max
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g2_fast \
      --all-probe --seeds 111 --max-train-samples 15000 --max-variants 3
    ;;
  full57)
    python scripts/run_full57.py --config "$CFG" --model axion --run-id axion_full57
    ;;
  *)
    echo "Unknown mode: $MODE (smoke|classical|g2|g2-fast|full57)" >&2
    exit 2
    ;;
esac

echo "== Done mode=$MODE"
