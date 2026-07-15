#!/usr/bin/env bash
# GATE v2: selective train-only ensemble on proven winner datasets only.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

if "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  HW=(--hardware 12gb)
else
  HW=()
fi

OUT_DIR="${OUT_DIR:-results/adadae_v51_gate/metrics}"
mkdir -p "$OUT_DIR"

# v5 postmortem: only unsup winners; exclude cover/celeba/CIFAR10/SVHN/Imdb semi
UNSUP_DATASETS=(speech ALOI optdigits)
SEMI_DATASETS=()
if [[ "${GATE_SEMI_SPEECH:-0}" == "1" ]]; then
  SEMI_DATASETS=(speech)
fi

FAILURES=0
run_job() {
  local ds="$1" setting="$2" seed="$3"
  if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
    --config configs/default_gpu.yaml \
    "${HW[@]}" \
    --dataset "$ds" \
    --setting "$setting" \
    --seed "$seed" \
    --override '{"adadae":{"policy":"routed","use_gate":true},"train":{"epochs":100}}' \
    --out-dir "$OUT_DIR"; then
    FAILURES=$((FAILURES + 1))
  fi
}

echo "=== GATE v2 (winner-take-all, real DDAE baseline) ==="
for ds in "${UNSUP_DATASETS[@]}"; do
  for seed in 111 222 333 444 555; do
    run_job "$ds" unsupervised "$seed"
  done
done
for ds in "${SEMI_DATASETS[@]}"; do
  for seed in 111 222 333 444 555; do
    run_job "$ds" semi-supervised "$seed"
  done
done

N=$("$PYTHON" -c "import json; from pathlib import Path; p=Path('$OUT_DIR/completed.json'); print(len(json.loads(p.read_text())['completed']) if p.exists() else 0)" 2>/dev/null || echo 0)
EXPECTED=$(( ${#UNSUP_DATASETS[@]} * 5 + ${#SEMI_DATASETS[@]} * 5 ))
echo "GATE v2 complete: $N/$EXPECTED jobs -> $OUT_DIR (failures=$FAILURES)"
if [[ "$N" -lt $(( EXPECTED / 2 )) ]]; then
  echo "ERROR: GATE job count too low"
  exit 1
fi
