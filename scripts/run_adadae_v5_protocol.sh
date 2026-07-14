#!/usr/bin/env bash
# AdaDDAE v5 full protocol: Phase0 eval -> Phase1 leftovers -> DAMP -> MCE -> SMC -> GATE -> merge -> gates.
# Usage:
#   bash scripts/run_adadae_v5_protocol.sh [phase0|phase1|damp|mce|smc|gate|merge|gates|thesis|all]
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
  echo "NOTE: No CUDA — CPU profile; GPU tracks skipped unless forced"
fi

MODE="${1:-all}"
HYBRID_OUT="${HYBRID_OUT:-results/adadae_v5_hybrid/metrics/completed.json}"
HYBRID_DIR="$(dirname "$(dirname "$HYBRID_OUT")")"
V41_BASE="${V41_BASE:-results/adadae_v41_hybrid/metrics/completed.json}"

run_phase0() {
  echo "=== Phase 0: eval contract + baseline freeze ==="
  "$PYTHON" scripts/compute_benchmark_tiers.py \
    --completed "$V41_BASE" \
    --out results/adadae_v5/thesis/eval_contract.json
  "$PYTHON" scripts/validate_gates.py \
    --completed "$V41_BASE" \
    --v5 \
    --v41-ref "$V41_BASE" \
    --out results/adadae_v5/thesis/baseline_v41_gates.json || true
}

run_phase1() {
  echo "=== Phase 1: routing leftovers (v4.1 protocol merge + semi reruns) ==="
  if [[ -f scripts/run_adadae_v41_protocol.sh ]]; then
    bash scripts/run_adadae_v41_protocol.sh phase0 || true
    if [[ "${SKIP_GPU:-0}" != "1" ]] && [[ ${#HW[@]} -gt 0 ]]; then
      bash scripts/run_adadae_v41_protocol.sh semi-reruns || true
      bash scripts/run_adadae_v41_protocol.sh bisect-fallback || true
    fi
    HYBRID_OUT=results/adadae_v5_phase1/metrics/completed.json \
      bash scripts/run_adadae_v41_protocol.sh merge || true
  else
    cp -f "$V41_BASE" results/adadae_v5_phase1/metrics/completed.json 2>/dev/null || \
      mkdir -p results/adadae_v5_phase1/metrics && cp -f "$V41_BASE" results/adadae_v5_phase1/metrics/completed.json
  fi
}

run_damp() {
  echo "=== Phase 2: Train DAMP meta-policy ==="
  "$PYTHON" scripts/train_damp.py
}

run_mce() {
  echo "=== Phase 3: MCE track (NLP/CV tails) ==="
  MCE_DATASETS=(ALOI celeba CIFAR10 SVHN speech Imdb Agnews Amazon Yelp 20newsgroups optdigits cover)
  for ds in "${MCE_DATASETS[@]}"; do
    for setting in unsupervised semi-supervised; do
      for seed in 111 222 333 444 555; do
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
          --config configs/adadae_v5_mce.yaml \
          "${HW[@]}" \
          --dataset "$ds" \
          --setting "$setting" \
          --seed "$seed" \
          --override "{\"adadae\":{\"use_mce\":true,\"mce_modality\":\"$( [[ \"$ds\" =~ ^(Imdb|Agnews|Amazon|Yelp|20newsgroups)$ ]] && echo nlp || echo cv )\"}}" \
          --out-dir results/adadae_v5_mce/metrics || true
      done
    done
  done
}

run_smc() {
  echo "=== Phase 4: SMC ablation (semi tails) ==="
  SMC_DATASETS=(glass vertebral Waveform speech)
  for ds in "${SMC_DATASETS[@]}"; do
    for seed in 111 222 333 444 555; do
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/adadae_v5_smc.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting semi-supervised \
        --seed "$seed" \
        --override '{"adadae":{"fusion_mode":"smc","use_multiview":true}}' \
        --out-dir results/adadae_v5_smc/metrics || true
    done
  done
}

run_gate() {
  echo "=== Phase 5: GATE ensemble ==="
  bash scripts/run_gate_protocol.sh
}

run_merge() {
  echo "=== Phase 6: Merge v5 hybrid -> $HYBRID_OUT ==="
  PHASE1_PATCH="results/adadae_v5_phase1/metrics/completed.json"
  [[ -f "$PHASE1_PATCH" ]] || PHASE1_PATCH="$V41_BASE"
  "$PYTHON" scripts/build_v5_hybrid.py \
    --base "$V41_BASE" \
    --phase1-patch "$PHASE1_PATCH" \
    --mce-patch results/adadae_v5_mce/metrics/completed.json \
    --smc-patch results/adadae_v5_smc/metrics/completed.json \
    --gate-patch results/adadae_v5_gate/metrics/completed.json \
    --out "$HYBRID_OUT" \
    --force
}

run_gates() {
  "$PYTHON" scripts/validate_gates.py \
    --completed "$HYBRID_OUT" \
    --v5 \
    --v41-ref "$V41_BASE" \
    --compare "$HYBRID_DIR/thesis/compare_to_ddae.json" \
    --out "$HYBRID_DIR/thesis/gates.json"
}

run_thesis() {
  "$PYTHON" scripts/compare_to_ddae.py \
    --completed "$HYBRID_OUT" \
    --out-dir "$HYBRID_DIR/thesis"
  "$PYTHON" scripts/generate_hybrid_thesis.py \
    --hybrid-dir "$HYBRID_DIR"
  "$PYTHON" scripts/compute_benchmark_tiers.py \
    --completed "$HYBRID_OUT" \
    --out "$HYBRID_DIR/thesis/eval_contract.json"
}

case "$MODE" in
  phase0) run_phase0 ;;
  phase1) run_phase1 ;;
  damp) run_damp ;;
  mce) run_mce ;;
  smc) run_smc ;;
  gate) run_gate ;;
  merge) run_merge ;;
  gates) run_gates ;;
  thesis) run_thesis ;;
  all)
    run_phase0
    run_phase1
    run_damp
    if [[ "${SKIP_GPU:-0}" != "1" ]] && [[ ${#HW[@]} -gt 0 ]]; then
      run_mce
      run_smc
      run_gate
    else
      echo "SKIP_GPU=1 or no CUDA — skipping MCE/SMC/GATE GPU tracks"
    fi
    run_merge
    run_thesis
    run_gates
    ;;
  local)
    run_phase0
    run_phase1
    run_damp
    run_merge
    run_thesis
    run_gates
    ;;
  *)
    echo "Usage: $0 [phase0|phase1|damp|mce|smc|gate|merge|gates|thesis|all|local]"
    exit 1
    ;;
esac
