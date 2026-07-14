#!/usr/bin/env bash
# Smoke gates for AdaDDAE v2 (GPU: --hardware 12gb; CPU laptop: omit HW flag).
# Usage: bash scripts/vast_adadae_v2_smoke.sh [12gb]
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
HW=()
if [[ "${1:-}" == "12gb" ]] || [[ "${1:-}" == "8gb" ]] || [[ "${1:-}" == "16gb" ]]; then
  HW=(--hardware "$1")
fi

echo "=== 1/4 Unsupervised SSTS: cardio ==="
python scripts/run_one.py \
  --config configs/adadae_unsup_ssts.yaml \
  "${HW[@]}" \
  --dataset cardio \
  --setting unsupervised \
  --seed 111

echo "=== 2/4 Unsupervised SSTS: ALOI ==="
python scripts/run_one.py \
  --config configs/adadae_unsup_ssts.yaml \
  "${HW[@]}" \
  --dataset ALOI \
  --setting unsupervised \
  --seed 111

echo "=== 3/4 Semi CV/NLP: Imdb ==="
python scripts/run_one.py \
  --config configs/adadae_semi_cvnlp.yaml \
  "${HW[@]}" \
  --dataset Imdb \
  --setting semi-supervised \
  --seed 111

echo "=== 4/4 Semi CV/NLP: SVHN ==="
python scripts/run_one.py \
  --config configs/adadae_semi_cvnlp.yaml \
  "${HW[@]}" \
  --dataset SVHN \
  --setting semi-supervised \
  --seed 111

echo ""
echo "Smoke done. Compare to backup/ddae_baseline_570/metrics/*__*__111.json"
echo "Full protocol: bash scripts/run_adadae_v2_protocol.sh"
