#!/usr/bin/env bash
# AdaDDAE v4.1 protocol: safe merge + full bisect + routing audit + v4.1 hybrid.
# Usage:
#   tmux new -s adadae_v41
#   bash scripts/run_adadae_v41_protocol.sh [phase0|bisect|merge|gates|full|all]
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
HYBRID_OUT="${HYBRID_OUT:-results/adadae_v41_hybrid/metrics/completed.json}"
HYBRID_DIR="$(dirname "$(dirname "$HYBRID_OUT")")"

run_safe_patches() {
  echo "=== v4.1 safe patches (fallback layer + robust semi + regression guard) ==="
  "$PYTHON" scripts/build_v31_patches.py \
    --regression-guard \
    --winner-mode robust \
    --bisect-matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-out results/adadae_v41_unsup/metrics/completed.json \
    --semi-out results/adadae_v41_semi_tail/metrics/completed.json
}

run_routing_rules() {
  echo "=== Build meta routing rules (category-guarded) ==="
  "$PYTHON" scripts/build_routing_rules.py \
    --semi-matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-matrix results/thesis/v4_unsup_bisect_matrix.csv
}

run_promote() {
  echo "=== Promote bisect winners to policy_exceptions.yaml ==="
  "$PYTHON" scripts/promote_bisect_policies.py \
    --winner-mode robust \
    --matrix results/thesis/v31_semi_tail_matrix.csv \
    --unsup-matrix results/thesis/v4_unsup_bisect_matrix.csv
}

run_audit_routing() {
  echo "=== Audit routing (G8) ==="
  "$PYTHON" scripts/audit_routing.py
}

run_unsup_nlp_bisect() {
  echo "=== Track 2: Unsup NLP bisect (5 datasets x 6 candidates x 5 seeds) ==="
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
  echo "=== Track 3: Unsup classical gap bisect (12 datasets) ==="
  "$PYTHON" scripts/v3_hard_bisect.py \
    "${HW[@]}" \
    --config configs/ablation_ladder.yaml \
    --unsup-classical-gap \
    --skip-ablations \
    --epochs 100 \
    --seeds 111 222 333 444 555 \
    --out results/thesis/v4_unsup_bisect_matrix.csv
}

run_unsup_fallback_bisect() {
  echo "=== Track 4: Unsup fallback bisect (14 classical datasets) ==="
  "$PYTHON" scripts/v3_hard_bisect.py \
    "${HW[@]}" \
    --config configs/ablation_ladder.yaml \
    --unsup-fallback-bisect \
    --skip-ablations \
    --epochs 100 \
    --seeds 111 222 333 444 555 \
    --out results/thesis/v41_unsup_fallback_bisect_matrix.csv
}

run_semi_reruns() {
  echo "=== Track 5: Semi tail robust reruns (glass, vertebral, Wilt, Waveform) ==="
  for ds in glass vertebral Wilt Waveform; do
    for seed in 111 222 333 444 555; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/adadae_v31_semi_tail.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting semi-supervised \
        --seed "$seed" \
        --override '{"adadae":{"policy":"routed"},"train":{"epochs":100}}'
    done
  done
}

run_smoke_vus() {
  echo "=== Track 6: VUS smoke on yeast + PageBlocks (unsup_classical_plus) ==="
  for ds in yeast PageBlocks; do
    for seed in 111 222 333; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/adadae_v4_unsup.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting unsupervised \
        --seed "$seed" \
        --override '{"adadae":{"policy":"unsup_classical_plus"},"train":{"epochs":100}}'
    done
  done
}

run_merge() {
  echo "=== Merge v4.1 hybrid -> $HYBRID_OUT ==="
  run_safe_patches
  if [[ -f results/thesis/v31_semi_tail_matrix.csv ]]; then
    run_routing_rules
    run_promote
  fi
  run_audit_routing

  "$PYTHON" scripts/build_v4_hybrid.py \
    --unsup results/adadae_unsup_ssts/metrics/completed.json \
    --unsup-fallback-patch results/adadae_v31_unsup/metrics/completed.json \
    --unsup-patch results/adadae_v41_unsup/metrics/completed.json \
    --semi-tail-patch results/adadae_v41_semi_tail/metrics/completed.json \
    --min-semi-pr 61.36 \
    --min-unsup-pr 36.77 \
    --out "$HYBRID_OUT"

  "$PYTHON" scripts/compare_to_ddae.py \
    --completed "$HYBRID_OUT" \
    --out-dir "$HYBRID_DIR/thesis"

  "$PYTHON" scripts/generate_hybrid_thesis.py \
    --hybrid-dir "$HYBRID_DIR"
}

run_gates() {
  "$PYTHON" scripts/validate_gates.py \
    --completed "$HYBRID_OUT" \
    --v31-ref results/adadae_v31_hybrid/metrics/completed.json \
    --compare "$HYBRID_DIR/thesis/compare_to_ddae.json" \
    --out "$HYBRID_DIR/thesis/gates.json"
}

case "$MODE" in
  phase0)
    run_safe_patches
    run_routing_rules
    run_promote
    run_audit_routing
    ;;
  bisect-nlp) run_unsup_nlp_bisect ;;
  bisect-classical) run_unsup_classical_bisect ;;
  bisect-fallback) run_unsup_fallback_bisect ;;
  semi-reruns) run_semi_reruns ;;
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
    run_unsup_fallback_bisect
    run_semi_reruns
    run_smoke_vus
    run_merge
    run_gates
    ;;
  *)
    echo "Usage: $0 [phase0|bisect-nlp|bisect-classical|bisect-fallback|semi-reruns|smoke-vus|merge|gates|all|full]"
    exit 1
    ;;
esac
