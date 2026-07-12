#!/usr/bin/env bash
# Bootstrap AdaDDAE on a Vast.ai GPU instance (auto-detects 8/12/16 GB profile)
set -euo pipefail
cd "$(dirname "$0")/.."

HARDWARE="${1:-}"

echo "=== GPU check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

echo "=== Python venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt

echo "=== Hardware profile ==="
if [[ -z "$HARDWARE" ]]; then
  python scripts/detect_hardware.py
  HARDWARE="$(python scripts/detect_hardware.py | awk -F= '/^suggested_profile=/{print $2}' | sed 's/\.yaml$//')"
fi
HW_ARG=()
if [[ -n "${HARDWARE:-}" ]]; then
  HW_ARG=(--hardware "$HARDWARE")
  echo "Using hardware profile: $HARDWARE"
fi

echo "=== CUDA / PyTorch ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("bf16", torch.cuda.is_bf16_supported())
    torch.cuda.reset_peak_memory_stats()
PY

echo "=== Smoke test (cardio, semi-supervised) ==="
python scripts/run_one.py \
  --config configs/default_gpu.yaml \
  --dataset cardio \
  --setting semi-supervised \
  --seed 111 \
  "${HW_ARG[@]}"

echo "=== Peak VRAM (if CUDA) ==="
python - <<'PY'
import torch
if torch.cuda.is_available():
    mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"peak_vram_mb={mb:.1f}")
PY

echo "Setup OK. Next steps:"
echo "  bash scripts/vast_smoke.sh ${HARDWARE:-}"
echo "  python scripts/run_full_protocol.py --config configs/default_gpu.yaml ${HW_ARG[*]}"
