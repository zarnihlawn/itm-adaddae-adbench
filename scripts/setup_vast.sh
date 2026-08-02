#!/usr/bin/env bash
# Bootstrap AdaDDAE on a Vast.ai GPU instance (auto-detects 8/12/16 GB profile)
set -euo pipefail
cd "$(dirname "$0")/.."

HARDWARE="${1:-}"

echo "=== GPU check ==="
nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

echo "=== Python venv ==="
deactivate 2>/dev/null || true
unset PYTHONHOME PYTHONPATH
rm -rf .venv
if ! python3 -c "import subprocess, _posixsubprocess" 2>/dev/null; then
  echo "System Python incomplete — installing python3-venv / full stdlib"
  apt-get update -y
  apt-get install -y python3 python3-venv python3-pip python3-full python3-dev || true
  apt-get install --reinstall -y python3 python3.12-minimal libpython3.12-stdlib || true
fi
if ! python3 -m venv .venv; then
  echo "ensurepip failed — creating venv without pip and bootstrapping"
  rm -rf .venv
  python3 -m venv --without-pip .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
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
