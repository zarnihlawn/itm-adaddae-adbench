#!/usr/bin/env bash
# AdaDDAE v5.1 protocol: guarded merge, MCE/SMC/GATE v2, beat v4.1.
# Usage:
#   bash scripts/run_adadae_v51_protocol.sh [phase0|phase1|damp|mce|smc|gate|merge|gates|thesis|all|vast-all]
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
  HAS_CUDA=1
else
  HW=()
  HAS_CUDA=0
  echo "NOTE: No CUDA — GPU tracks will fail unless skipped"
fi

MODE="${1:-all}"
HYBRID_OUT="${HYBRID_OUT:-results/adadae_v51_hybrid/metrics/completed.json}"
HYBRID_DIR="$(dirname "$(dirname "$HYBRID_OUT")")"
V41_BASE="${V41_BASE:-results/adadae_v41_hybrid/metrics/completed.json}"
USE_GATE="${USE_GATE:-0}"

mce_modality() {
  case "$1" in
    Imdb|Agnews|Amazon|Yelp|20newsgroups) echo nlp ;;
    ALOI|celeba|CIFAR10|SVHN|speech) echo cv ;;
    optdigits|cover) echo classical ;;
    *) echo classical ;;
  esac
}

run_phase0() {
  echo "=== Phase 0: eval contract ==="
  "$PYTHON" scripts/compute_benchmark_tiers.py \
    --completed "$V41_BASE" \
    --out results/adadae_v51/thesis/eval_contract.json
}

run_phase1() {
  echo "=== Phase 1: semi reruns + fallback bisect ==="
  if [[ "$HAS_CUDA" -eq 1 ]]; then
    bash scripts/run_adadae_v41_protocol.sh semi-reruns
    bash scripts/run_adadae_v41_protocol.sh bisect-fallback
  else
    echo "SKIP: semi-reruns/bisect-fallback need GPU"
  fi
  mkdir -p results/adadae_v51_phase1/metrics
  if [[ -f results/adadae_v41_semi_tail/metrics/completed.json ]] || [[ -f results/adadae_v41_unsup/metrics/completed.json ]]; then
    HYBRID_OUT=results/adadae_v51_phase1/metrics/completed.json \
      bash scripts/run_adadae_v41_protocol.sh merge
  else
    cp -f "$V41_BASE" results/adadae_v51_phase1/metrics/completed.json
  fi
  "$PYTHON" scripts/check_v51_track.py phase1 --min-frac 0.9 || true
}

run_damp() {
  echo "=== DAMP retrain ==="
  "$PYTHON" scripts/train_damp.py
}

run_mce() {
  echo "=== MCE track (v5.1 modality map) ==="
  OUT=results/adadae_v51_mce/metrics
  mkdir -p "$OUT"
  FAILURES=0
  MCE_DATASETS=(ALOI celeba CIFAR10 SVHN speech Imdb Agnews Amazon Yelp 20newsgroups optdigits cover)
  NLP_DS="Imdb Agnews Amazon Yelp 20newsgroups"
  for ds in "${MCE_DATASETS[@]}"; do
    MOD="$(mce_modality "$ds")"
    for setting in unsupervised semi-supervised; do
      if [[ "$setting" == "semi-supervised" ]] && echo "$NLP_DS" | grep -qw "$ds"; then
        continue
      fi
      for seed in 111 222 333 444 555; do
        if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
          --config configs/adadae_v5_mce.yaml \
          "${HW[@]}" \
          --dataset "$ds" \
          --setting "$setting" \
          --seed "$seed" \
          --override "{\"adadae\":{\"use_mce\":true,\"mce_modality\":\"$MOD\",\"mce_block_semi_nlp\":true}}" \
          --out-dir "$OUT"; then
          FAILURES=$((FAILURES + 1))
        fi
      done
    done
  done
  echo "MCE failures: $FAILURES"
  "$PYTHON" scripts/check_v51_track.py mce --min-frac 0.5
}

run_smc() {
  echo "=== SMC track (semi tails) ==="
  OUT=results/adadae_v51_smc/metrics
  mkdir -p "$OUT"
  FAILURES=0
  for ds in glass vertebral Waveform speech; do
    for seed in 111 222 333 444 555; do
      if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/run_one.py \
        --config configs/adadae_v5_smc.yaml \
        "${HW[@]}" \
        --dataset "$ds" \
        --setting semi-supervised \
        --seed "$seed" \
        --override '{"adadae":{"fusion_mode":"smc","use_multiview":true}}' \
        --out-dir "$OUT"; then
        FAILURES=$((FAILURES + 1))
      fi
    done
  done
  echo "SMC failures: $FAILURES"
  "$PYTHON" scripts/check_v51_track.py smc --min-frac 0.5
}

run_gate() {
  echo "=== GATE v2 (USE_GATE=${USE_GATE}) ==="
  OUT_DIR=results/adadae_v51_gate/metrics bash scripts/run_gate_protocol.sh
  "$PYTHON" scripts/check_v51_track.py gate --min-frac 0.5
}

run_merge() {
  echo "=== Guarded merge -> $HYBRID_OUT ==="
  PHASE1="results/adadae_v51_phase1/metrics/completed.json"
  [[ -f "$PHASE1" ]] || PHASE1="$V41_BASE"
  MERGE_ARGS=(
    --base "$V41_BASE"
    --phase1-patch "$PHASE1"
    --mce-patch results/adadae_v51_mce/metrics/completed.json
    --smc-patch results/adadae_v51_smc/metrics/completed.json
    --guard-epsilon 0.1
    --strict-beat-v41
    --out "$HYBRID_OUT"
  )
  if [[ "$USE_GATE" == "1" ]] && [[ -f results/adadae_v51_gate/metrics/completed.json ]]; then
    MERGE_ARGS+=(--use-gate)
  fi
  if ! "$PYTHON" scripts/build_v51_hybrid.py "${MERGE_ARGS[@]}"; then
    echo "Strict beat failed — retry without gate patch"
    "$PYTHON" scripts/build_v51_hybrid.py \
      --base "$V41_BASE" \
      --phase1-patch "$PHASE1" \
      --mce-patch results/adadae_v51_mce/metrics/completed.json \
      --smc-patch results/adadae_v51_smc/metrics/completed.json \
      --guard-epsilon 0.1 \
      --out "$HYBRID_OUT" \
      --force
  fi
}

run_gates() {
  "$PYTHON" scripts/validate_gates.py \
    --completed "$HYBRID_OUT" \
    --v5 \
    --v5-strict \
    --v41-ref "$V41_BASE" \
    --merge-audit "$HYBRID_DIR/thesis/merge_audit.json" \
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
  gate) USE_GATE=1 run_gate ;;
  merge) run_merge ;;
  gates) run_gates ;;
  thesis) run_thesis ;;
  all)
    run_phase0
    run_phase1
    run_damp
    if [[ "$HAS_CUDA" -eq 1 ]]; then
      run_mce
      run_smc
      USE_GATE=1 run_gate
    else
      echo "No CUDA — skipping GPU tracks"
    fi
    run_merge
    run_thesis
    run_gates
    ;;
  vast-all)
    run_phase0
    run_phase1
    run_damp
    run_mce
    run_smc
    USE_GATE=1 run_gate
    run_merge
    run_thesis
    run_gates
    ;;
  local-merge)
    run_phase0
    run_damp
    USE_GATE=1 run_merge
    run_thesis
    run_gates
    ;;
  *)
    echo "Usage: $0 [phase0|phase1|damp|mce|smc|gate|merge|gates|thesis|all|vast-all|local-merge]"
    exit 1
    ;;
esac
