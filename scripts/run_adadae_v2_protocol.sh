#!/usr/bin/env bash
# Production AdaDDAE v2 hybrid protocol (run in tmux on Vast RTX 3060 12GB).
# Usage:
#   tmux new -s adadae_v2
#   bash scripts/run_adadae_v2_protocol.sh [unsup|semi|merge|all]
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
HW=(--hardware 12gb)
MODE="${1:-all}"

CV_NLP_DATASETS=(
  CIFAR10 FashionMNIST MNIST-C MVTec-AD SVHN
  Agnews Amazon Imdb Yelp 20newsgroups
)

run_unsup() {
  echo "=== Unsupervised 285 jobs (LF-DANC + MANS + SSTS) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_full_protocol.py \
    --config configs/adadae_unsup_ssts.yaml \
    "${HW[@]}" \
    --settings unsupervised
}

run_semi_cvnlp() {
  echo "=== Semi CV/NLP 50 jobs (FTP + light TAPS) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_full_protocol.py \
    --config configs/adadae_semi_cvnlp.yaml \
    "${HW[@]}" \
    --settings semi-supervised \
    --datasets "${CV_NLP_DATASETS[@]}"
}

run_merge() {
  echo "=== Merge hybrid + compare to DDAE ==="
  python scripts/merge_completed.py \
    --semi-classical backup/ddae_baseline_570/metrics/completed.json \
    --semi-cvnlp results/adadae_semi_cvnlp/metrics/completed.json \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --out results/adadae_v2_hybrid/metrics/completed.json \
    --copy-metrics

  python scripts/compare_to_ddae.py \
    --completed results/adadae_v2_hybrid/metrics/completed.json \
    --out-dir results/adadae_v2_hybrid/thesis

  python scripts/generate_hybrid_thesis.py
}

case "$MODE" in
  unsup) run_unsup ;;
  semi) run_semi_cvnlp ;;
  merge) run_merge ;;
  all)
    run_unsup
    run_semi_cvnlp
    run_merge
    ;;
  *)
    echo "Usage: $0 [unsup|semi|merge|all]"
    exit 1
    ;;
esac
