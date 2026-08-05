# Final run + resume — Vast AdaDDAE-PER

**Offer:** Ohio, US · 1× RTX 4060 Ti **16 GB** · AMD EPYC 7K62 · **16 CPU** · **~43 GB RAM** · ~$0.152/hr  
**Hardware flag everywhere:** `16gb`  
**Canonical protocol:** [`FINAL_RUN.md`](FINAL_RUN.md) — **AdaDDAE-PER only** (frozen v2→v5.1 rules)

Replace `PORT` / `sshN` with values from the Vast instance SSH button.

---

## 0. SSH in

```bash
ssh -p PORT root@sshN.vast.ai
```

Confirm disk (**must be ≥ ~100 GB**, not 10–16 GB):

```bash
df -h / /workspace
nvidia-smi
nproc
free -h
```

If `/workspace` shows **~10G or ~16G** total → **destroy and re-rent** with Container Size ≥ 100 GB.

---

## 1. Layout, packages, threads, tmux

```bash
mkdir -p /workspace/ITM
cd /workspace/ITM

apt-get update -y && apt-get install -y tmux git rsync \
  python3 python3-venv python3-pip python3-full python3-dev

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cat >> ~/.bashrc <<'EOF'
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
EOF

tmux new -s adadae
```

---

## 2. Clone ADBench + project

```bash
cd /workspace/ITM

git clone --depth 1 https://github.com/Minqi824/ADBench.git ADBench
rm -rf ADBench/adbench/datasets/CV_by_ViT \
       ADBench/adbench/datasets/NLP_by_RoBERTa

git clone https://github.com/zarnihlawn/itm-adaddae.git project
cd /workspace/ITM/project
git checkout main
git pull origin main
mkdir -p results
```

If you already have local `results/ddae_baseline_valstop` (570/570), rsync it onto the box before training.

---

## 3. Python GPU env

```bash
cd /workspace/ITM/project
deactivate 2>/dev/null || true
unset PYTHONHOME PYTHONPATH
rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

Confirm CUDA, then:

```bash
python scripts/detect_hardware.py
python scripts/assert_final_config.py --config configs/adadae_per.yaml
python scripts/assert_final_config.py --config configs/baselines_ddae_valstop.yaml --allow-nonfinal-run-id
bash scripts/run_adadae_per_protocol.sh dump_routing
```

Every assert must print **`INTEGRITY OK`**.

---

## 4. Production sequence

```bash
cd /workspace/ITM/project
source .venv/bin/activate

bash scripts/run_adadae_per_protocol.sh smoke 16gb
bash scripts/run_adadae_per_protocol.sh ddae 16gb    # skip if already 570/570
bash scripts/run_adadae_per_protocol.sh final 16gb   # THE one model, full 570
bash scripts/run_adadae_per_protocol.sh compare
bash scripts/run_adadae_per_protocol.sh gates
```

Or:

```bash
bash scripts/run_adadae_per_protocol.sh all 16gb
```

**Ship gate:** `results/adadae_per/thesis/integrity_gates.json` → `all_pass: true` (`G_paper_both`).

---

## 5. Resume

```bash
tmux attach -t adadae || tmux new -s adadae
cd /workspace/ITM/project && source .venv/bin/activate && source ~/.bashrc
bash scripts/run_adadae_per_protocol.sh final 16gb
```

### Progress

```bash
python - <<'PY'
import json
from pathlib import Path
for t in ["ddae_baseline_valstop", "adadae_per"]:
    p = Path(f"results/{t}/metrics/completed.json")
    if not p.exists():
        print(f"{t}: MISSING"); continue
    print(f"{t}: {len(json.loads(p.read_text()).get('completed', {}))}/570")
PY
```

---

## 6. Pull results

```bash
rsync -avz --progress -e 'ssh -p PORT' \
  root@sshN.vast.ai:/workspace/ITM/project/results/ \
  ./results/
```

---

## Quick map

| Step | Command |
|------|---------|
| Dump routing | `bash scripts/run_adadae_per_protocol.sh dump_routing` |
| Smoke | `… smoke 16gb` |
| Fair DDAE | `… ddae 16gb` |
| **PER 570** | `… final 16gb` |
| Compare + paper gate | `… compare` then `… gates` |
| Hardware | Always **`16gb`** on this box |
