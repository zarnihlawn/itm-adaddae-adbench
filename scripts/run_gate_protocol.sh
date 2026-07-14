#!/usr/bin/env bash
# GATE protocol: train-only ensemble reruns on backup-loser datasets.
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

OUT_DIR="${OUT_DIR:-results/adadae_v5_gate/metrics}"
mkdir -p "$OUT_DIR"

# Datasets where v4.1 loses to DDAE backup by >0.5pp (unsup focus)
LOSER_DATASETS=(optdigits ALOI speech Imdb celeba cover CIFAR10 SVHN)

echo "=== GATE ensemble reruns (${#LOSER_DATASETS[@]} datasets x 2 settings x 5 seeds) ==="
for ds in "${LOSER_DATASETS[@]}"; do
  for setting in unsupervised semi-supervised; do
    for seed in 111 222 333 444 555; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/default_gpu.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting "$setting" \
        --seed "$seed" \
        --override '{"adadae":{"policy":"routed","use_gate":true},"train":{"epochs":100}}' \
        --out-dir "$OUT_DIR" || true
    done
  done
done

echo "GATE protocol complete -> $OUT_DIR"
