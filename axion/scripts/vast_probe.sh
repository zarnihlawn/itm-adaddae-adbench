#!/usr/bin/env bash
# Run AXION probes on Vast GPU.
# Usage:
#   bash scripts/vast_probe.sh smoke
#   bash scripts/vast_probe.sh g4-classical configs/gpu.yaml
#   bash scripts/vast_probe.sh g4 configs/gpu.yaml          # requires classical gate
#   bash scripts/vast_probe.sh g4-auto configs/gpu.yaml     # classical → gate → full
#   bash scripts/vast_probe.sh g3-classical configs/gpu.yaml  # legacy
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

require_classical_gate() {
  local run_id="${1:-axion_g4_classical}"
  echo "== Checking classical gate: $run_id (semi PR ≥ 60)"
  python scripts/check_classical_gate.py --config "$CFG" --run-id "$run_id" --min-semi-pr 60
}

case "$MODE" in
  smoke)
    python scripts/run_probe.py --config "$CFG" --smoke --model axion --loop-log --run-id axion_smoke
    ;;
  classical)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_classical \
      --datasets breastw cardio glass thyroid satimage-2 cover fraud backdoor \
      --seeds 111 222 333 \
      --max-train-samples 20000
    ;;
  g2)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g2 \
      --all-probe --max-train-samples 20000 --max-variants 0
    ;;
  g2-fast)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g2_fast \
      --all-probe --seeds 111 --max-train-samples 15000 --max-variants 3
    ;;
  g3)
    echo "WARN: g3 is legacy; prefer g4 / g4-auto (hard classical gate)."
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g3 \
      --all-probe --max-train-samples 20000 --max-variants 0
    ;;
  g3-classical)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g3_classical \
      --datasets breastw cardio glass thyroid satimage-2 cover fraud backdoor \
      --seeds 111 \
      --max-train-samples 20000
    ;;
  g4-classical)
    # Cheap classical gate (8 × 2 × seed 111) — Phase 4 defaults
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g4_classical \
      --datasets breastw cardio glass thyroid satimage-2 cover fraud backdoor \
      --seeds 111 \
      --max-train-samples 20000
    echo "== Classical MACRO (read before full g4):"
    python scripts/check_classical_gate.py --config "$CFG" --run-id axion_g4_classical --min-semi-pr 60 || true
    ;;
  g4)
    # Full Phase-4 probe — HARD STOP unless classical gate passed
    require_classical_gate axion_g4_classical
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g4 \
      --all-probe --max-train-samples 20000 --max-variants 0
    ;;
  g4-auto)
    # One command: classical → hard gate → full (never burns GPU on semi<60)
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g4_classical \
      --datasets breastw cardio glass thyroid satimage-2 cover fraud backdoor \
      --seeds 111 \
      --max-train-samples 20000
    require_classical_gate axion_g4_classical
    python scripts/run_probe.py --config "$CFG" --model axion --loop-log --run-id axion_g4 \
      --all-probe --max-train-samples 20000 --max-variants 0
    ;;
  full57)
    require_classical_gate axion_g4_classical
    echo "== Also require full g4 margins before full57"
    python scripts/check_classical_gate.py --config "$CFG" --run-id axion_g4 --min-semi-pr 63.36 --require-margin \
      || { echo "Run g4 first and pass both margins before full57"; exit 2; }
    python scripts/run_full57.py --config "$CFG" --model axion --run-id axion_full57
    ;;
  *)
    echo "Unknown mode: $MODE (smoke|classical|g2|g2-fast|g3|g3-classical|g4-classical|g4|g4-auto|full57)" >&2
    exit 2
    ;;
esac

echo "== Done mode=$MODE"
