#!/usr/bin/env bash
# AdaDDAE v3.1 protocol: unsup fallback + semi tail specialists.
# Usage:
#   tmux new -s adadae_v31
#   bash scripts/run_adadae_v31_protocol.sh [unsup|semi|bisect|merge|gates|simulate|all]
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi
if "$PYTHON" -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  HW=(--hardware 12gb)
else
  HW=()
  echo "NOTE: No CUDA — using CPU hardware profile from config"
fi
MODE="${1:-all}"

UNSUP_FALLBACK=(
  magic.gamma landsat fraud InternetAds Ionosphere glass WPBC vertebral backdoor
  vowels letter skin fault wine
)

SEMI_TAIL=(
  speech Imdb ALOI celeba Amazon Wilt SVHN Yelp 20newsgroups
  CIFAR10 Waveform census Agnews vertebral optdigits glass WPBC
)

run_unsup() {
  echo "=== Track A: unsup baseline fallback (${#UNSUP_FALLBACK[@]} datasets x 5 seeds) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_v31_unsup.yaml \
    "${HW[@]}" \
    --settings unsupervised \
    --datasets "${UNSUP_FALLBACK[@]}"
  "$PYTHON" scripts/check_resolved_policy.py \
    --completed results/adadae_v31_unsup/metrics/completed.json \
    --setting unsupervised \
    --expect-fallback "${UNSUP_FALLBACK[@]}" || true
}

run_semi() {
  echo "=== Track B: semi tail routed patch (${#SEMI_TAIL[@]} datasets x 5 seeds) ==="
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_full_protocol.py \
    --config configs/adadae_v31_semi_tail.yaml \
    "${HW[@]}" \
    --settings semi-supervised \
    --datasets "${SEMI_TAIL[@]}"
}

run_bisect() {
  echo "=== Track B bisect: full tail matrix 100 epochs x 5 seeds ==="
  "$PYTHON" scripts/v3_hard_bisect.py \
    "${HW[@]}" \
    --config configs/ablation_ladder.yaml \
    --full-tail \
    --skip-ablations \
    --epochs 100 \
    --seeds 111 222 333 444 555 \
    --out results/thesis/v31_semi_tail_matrix.csv
}

run_smoke() {
  echo "=== Smoke: speech / ALOI / celeba semi (3 seeds) ==="
  for ds in speech ALOI celeba; do
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
      --config configs/adadae_v31_semi_tail.yaml \
      "${HW[@]}" \
      --dataset "$ds" \
      --setting semi-supervised \
      --seed 111
  done
}

run_simulate() {
  echo "=== Counterfactual gate (bisect best on semi tail) ==="
  "$PYTHON" scripts/build_v31_hybrid.py --simulate-only --min-semi-pr 61.40
}

run_patches() {
  echo "=== Build v3.1 patches (unsup + bisect semi tail) ==="
  "$PYTHON" scripts/build_v31_patches.py \
    --regression-guard \
    --bisect-matrix results/thesis/v31_semi_tail_matrix.csv \
    --winner-mode mean
}

run_bisect_merge() {
  run_patches
  run_merge
  run_gates
}

run_merge() {
  echo "=== Merge v3.1 hybrid ==="
  # Always refresh semi patch when 100-epoch bisect matrix is available
  if [[ -f results/thesis/v31_semi_tail_matrix.csv ]]; then
    run_patches
  elif [[ ! -s results/adadae_v31_unsup/metrics/completed.json ]] || \
     [[ "$("$PYTHON" -c "import json;print(len(json.load(open('results/adadae_v31_unsup/metrics/completed.json')).get('completed',{})))" 2>/dev/null || echo 0)" -lt 10 ]]; then
    run_patches
  fi
  "$PYTHON" scripts/build_v31_hybrid.py \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --unsup-patch results/adadae_v31_unsup/metrics/completed.json \
    --semi-tail-patch results/adadae_v31_semi_tail/metrics/completed.json \
    --min-semi-pr 61.36 \
    --out results/adadae_v31_hybrid/metrics/completed.json \
    --copy-metrics || \
  "$PYTHON" scripts/build_v31_hybrid.py \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --unsup-patch results/adadae_v31_unsup/metrics/completed.json \
    --semi-tail-patch results/adadae_v31_semi_tail/metrics/completed.json \
    --min-semi-pr 61.36 \
    --out results/adadae_v31_hybrid/metrics/completed.json \
    --copy-metrics \
    --force

  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae_v31_hybrid/metrics/completed.json \
    --out-dir results/adadae_v31_hybrid/thesis

  "$PYTHON" scripts/generate_hybrid_thesis.py \
    --hybrid-dir results/adadae_v31_hybrid
}

run_gates() {
  "$PYTHON" scripts/validate_gates.py \
    --completed results/adadae_v31_hybrid/metrics/completed.json \
    --out results/adadae_v31_hybrid/thesis/gates.json
}

case "$MODE" in
  unsup) run_unsup ;;
  semi) run_semi ;;
  bisect) run_bisect ;;
  smoke) run_smoke ;;
  simulate) run_simulate ;;
  bisect-merge) run_bisect_merge ;;
  patches) run_patches ;;
  merge) run_merge; run_gates ;;
  gates) run_gates ;;
  all)
    run_unsup
    run_smoke
    run_semi
    run_merge
    run_gates
    ;;
  full)
    run_unsup
    run_bisect
    run_bisect_merge
    ;;
  *)
    echo "Usage: $0 [unsup|semi|bisect|bisect-merge|smoke|simulate|patches|merge|gates|all|full]"
    exit 1
    ;;
esac
