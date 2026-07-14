#!/usr/bin/env bash
# AdaDDAE v4 protocol: safe merge + unsup bisect + meta routing + v4 hybrid.
# Usage:
#   tmux new -s adadae_v4
#   bash scripts/run_adadae_v4_protocol.sh [safe-merge|unsup-nlp|unsup-classical|smoke-vus|merge|gates|all|full]
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

run_safe_patches() {
  echo "=== v4 safe patches (regression guard + NLP semi freeze) ==="
  "$PYTHON" scripts/build_v31_patches.py \
    --regression-guard \
    --bisect-matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-out results/adadae_v4_unsup/metrics/completed.json \
    --semi-out results/adadae_v4_semi_tail/metrics/completed.json
}

run_routing_rules() {
  echo "=== Build meta routing rules ==="
  "$PYTHON" scripts/build_routing_rules.py \
    --semi-matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-matrix results/thesis/v4_unsup_bisect_matrix.csv
}

run_promote() {
  echo "=== Promote bisect winners to policy_exceptions.yaml ==="
  "$PYTHON" scripts/promote_bisect_policies.py \
    --matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-matrix results/thesis/v4_unsup_bisect_matrix.csv
}

run_unsup_nlp_bisect() {
  echo "=== Unsup NLP bisect (5 datasets x 5 candidates x 5 seeds) ==="
  "$PYTHON" scripts/v3_hard_bisect.py \
    "${HW[@]}" \
    --config configs/ablation_ladder.yaml \
    --unsup-nlp \
    --skip-ablations \
    --epochs 100 \
    --seeds 111 222 333 444 555 \
    --out results/thesis/v4_unsup_bisect_matrix.csv
}

run_unsup_classical_bisect() {
  echo "=== Unsup classical gap bisect (12 datasets) ==="
  "$PYTHON" scripts/v3_hard_bisect.py \
    "${HW[@]}" \
    --config configs/ablation_ladder.yaml \
    --unsup-classical-gap \
    --skip-ablations \
    --epochs 100 \
    --seeds 111 222 333 444 555 \
    --out results/thesis/v4_unsup_bisect_matrix.csv
}

run_smoke_vus() {
  echo "=== Smoke: unsup_classical_plus on cover / cardio / vowels (3 seeds) ==="
  for ds in cover cardio vowels; do
    for seed in 111 222 333; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/adadae_v4_unsup.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting unsupervised \
        --seed "$seed" \
        --override '{"adadae":{"policy":"routed"},"train":{"epochs":100}}'
    done
  done
}

run_merge() {
  echo "=== Merge v4 hybrid ==="
  run_safe_patches
  if [[ -f results/thesis/v31_semi_tail_matrix.csv ]]; then
    run_routing_rules || true
    run_promote || true
  fi
  "$PYTHON" scripts/build_v4_hybrid.py \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --unsup-patch results/adadae_v4_unsup/metrics/completed.json \
    --semi-tail-patch results/adadae_v4_semi_tail/metrics/completed.json \
    --min-semi-pr 61.36 \
    --min-unsup-pr 36.77 \
    --out results/adadae_v4_hybrid/metrics/completed.json \
    --copy-metrics || \
  "$PYTHON" scripts/build_v4_hybrid.py \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --unsup-patch results/adadae_v4_unsup/metrics/completed.json \
    --semi-tail-patch results/adadae_v4_semi_tail/metrics/completed.json \
    --min-semi-pr 61.36 \
    --min-unsup-pr 36.77 \
    --out results/adadae_v4_hybrid/metrics/completed.json \
    --copy-metrics \
    --force

  "$PYTHON" scripts/compare_to_ddae.py \
    --completed results/adadae_v4_hybrid/metrics/completed.json \
    --out-dir results/adadae_v4_hybrid/thesis

  "$PYTHON" scripts/generate_hybrid_thesis.py \
    --hybrid-dir results/adadae_v4_hybrid
}

run_gates() {
  "$PYTHON" scripts/validate_gates.py \
    --completed results/adadae_v4_hybrid/metrics/completed.json \
    --v31-ref results/adadae_v31_hybrid/metrics/completed.json \
    --out results/adadae_v4_hybrid/thesis/gates.json
}

case "$MODE" in
  safe-merge) run_safe_patches; run_merge; run_gates ;;
  safe-patches) run_safe_patches ;;
  routing) run_routing_rules ;;
  promote) run_promote ;;
  unsup-nlp) run_unsup_nlp_bisect ;;
  unsup-classical) run_unsup_classical_bisect ;;
  smoke-vus) run_smoke_vus ;;
  merge) run_merge; run_gates ;;
  gates) run_gates ;;
  all)
    run_safe_patches
    run_merge
    run_gates
    ;;
  full)
    run_unsup_nlp_bisect
    run_unsup_classical_bisect
    run_smoke_vus
    run_safe_patches
    run_routing_rules
    run_promote
    run_merge
    run_gates
    ;;
  *)
    echo "Usage: $0 [safe-merge|safe-patches|routing|promote|unsup-nlp|unsup-classical|smoke-vus|merge|gates|all|full]"
    exit 1
    ;;
esac
