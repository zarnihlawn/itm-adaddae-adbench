#!/usr/bin/env bash
# AdaDDAE v3 selective protocol (~100-130 jobs, not full 570).
# Usage:
#   tmux new -s adadae_v3
#   bash scripts/run_adadae_v3_protocol.sh [patch|merge|gates|all]
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
HW=(--hardware 12gb)
MODE="${1:-all}"

# Datasets that need v3 reruns (policy changes vs v2)
UNSUP_PATCH_DATASETS=(
  vowels letter skin fault wine glass
  Agnews Amazon Imdb Yelp 20newsgroups
)
SEMI_PATCH_DATASETS=(
  speech
  CIFAR10 FashionMNIST MNIST-C MVTec-AD SVHN
)

run_patch() {
  echo "=== v3 routed patch: unsup fallback + NLP baseline + semi specialists ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_full_protocol.py \
    --config configs/adadae_v3_hybrid.yaml \
    "${HW[@]}" \
    --settings unsupervised \
    --datasets "${UNSUP_PATCH_DATASETS[@]}"

  echo "=== v3 semi patch: speech specialist + CV FTP policy ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/run_full_protocol.py \
    --config configs/adadae_v3_hybrid.yaml \
    "${HW[@]}" \
    --settings semi-supervised \
    --datasets "${SEMI_PATCH_DATASETS[@]}"
}

run_merge() {
  echo "=== Merge v3 hybrid (oracle unsup + backup semi NLP + patch overrides) ==="
  python scripts/merge_completed.py \
    --semi-classical backup/ddae_baseline_570/metrics/completed.json \
    --semi-cvnlp results/adadae_semi_cvnlp/metrics/completed.json \
    --semi-cvnlp-source backup \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --patch results/adadae_v3_routed/metrics/completed.json \
    --out results/adadae_v3_hybrid/metrics/completed.json \
    --copy-metrics

  python scripts/compare_to_ddae.py \
    --completed results/adadae_v3_hybrid/metrics/completed.json \
    --out-dir results/adadae_v3_hybrid/thesis

  python scripts/validate_gates.py \
    --completed results/adadae_v3_hybrid/metrics/completed.json \
    --out results/adadae_v3_hybrid/thesis/gates.json

  python scripts/generate_hybrid_thesis.py \
    --hybrid-dir results/adadae_v3_hybrid
}

run_gates() {
  python scripts/validate_gates.py \
    --completed results/adadae_v3_hybrid/metrics/completed.json \
    --out results/adadae_v3_hybrid/thesis/gates.json
}

run_oracle_offline() {
  echo "=== Offline oracle-best hybrid (no GPU) ==="
  python scripts/build_oracle_hybrid.py \
    --out results/adadae_v3_hybrid/metrics/completed.json \
    --copy-metrics

  python scripts/compare_to_ddae.py \
    --completed results/adadae_v3_hybrid/metrics/completed.json \
    --out-dir results/adadae_v3_hybrid/thesis

  python scripts/validate_gates.py \
    --completed results/adadae_v3_hybrid/metrics/completed.json \
    --out results/adadae_v3_hybrid/thesis/gates.json || true
}

case "$MODE" in
  patch) run_patch ;;
  merge) run_merge ;;
  gates) run_gates ;;
  oracle) run_oracle_offline ;;
  all)
    run_patch
    run_merge
    ;;
  *)
    echo "Usage: $0 [patch|merge|gates|oracle|all]"
    exit 1
    ;;
esac
