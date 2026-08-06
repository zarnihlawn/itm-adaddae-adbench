#!/usr/bin/env bash
# Vast / GPU setup for AXION (ITM/project/axion — same git repo as project).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== AXION root: $ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip wheel
python -m pip install -r requirements.txt

# Prefer CUDA torch if nvidia-smi exists
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== NVIDIA GPU detected; ensuring CUDA torch"
  python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu124 || \
    python -m pip install -U torch
else
  echo "== No nvidia-smi; keeping CPU torch"
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import torch
from axion.paths import DEFAULT_ADBENCH_DATASETS, AXION_ROOT, ITM_ROOT
print("AXION_ROOT", AXION_ROOT)
print("ITM_ROOT", ITM_ROOT)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("adbench", DEFAULT_ADBENCH_DATASETS, "exists=", DEFAULT_ADBENCH_DATASETS.exists())
PY

python scripts/build_atlas.py
pytest -q
echo "== Setup OK. See VAST_RUN.md"
