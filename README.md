# AdaDDAE: Adaptive Diffusion-Scheduled Denoising Autoencoder

Master's thesis implementation for tabular anomaly detection on **ADBench** (57 datasets), designed to outperform **DDAE** (KDD 2025, [arXiv:2508.00758](https://arxiv.org/abs/2508.00758)).

## Novelty

AdaDDAE is a **named component framework** — each piece has an equation and an ablation step:

| Acronym | Component |
|---------|-----------|
| **FTP** | Leak-safe Feature Tuning Pipeline (PCA, scaler) |
| **LF-DANC** | Label-free Dataset-Adaptive Noise Controller |
| **MANS** | Manifold-Aligned Noise Schedule (\(\beta^*\), \(\tau^*\)) |
| **SSTS** | SNR-Stratified Timestep Selection (importance sampling) |
| **TAPS** | Timestep-Adaptive Pair Sampling (normal-only contrastive) |
| **VUS** | Variance-Uncertainty Score (4th fusion view) |
| **RDT** | Rejection-aware Diffusion Training (TabADM-inspired) |
| **DTE-View** | Diffusion Time Posterior Score ([ICLR 2024 DTE](https://arxiv.org/abs/2305.18593)) |

Full showcase: [`thesis/novelty.md`](thesis/novelty.md) · Equations: [`thesis/method.md`](thesis/method.md)

```bash
# Contribution waterfall (5 datasets × 2 settings)
python scripts/showcase_novelty.py --epochs 20

# Full ablation ladder with ΔPR-AUC CSV
python scripts/ablations.py --config configs/ablation_ladder.yaml

# Timing breakdown (ftp / train / score)
python scripts/profile_job.py
```

## Hardware profiles

| Profile | Config | VRAM soft limit | Machine |
|---------|--------|-----------------|---------|
| **CPU** | `configs/default.yaml` | — | Intel i5-12500H, 15 GiB RAM |
| **GPU 16 GB** (default) | `configs/default_gpu.yaml` | 14 GB | RTX 5070 Ti / 5060 Ti (Vast #44074990, #28576024) |
| **GPU 12 GB** | `default_gpu.yaml` + `--hardware 12gb` | 10 GB | RTX 5070 12 GB (Vast #43038090) |
| **GPU 8 GB** | `configs/default_gpu_8g.yaml` | 6.8 GB | RTX 5060 8 GB + 64 GB RAM (Vast #42076734) |

Hardware YAML files: `hardware_rtx5070ti.yaml`, `hardware_rtx5070_12g.yaml`, `hardware_rtx5070.yaml`.

Auto-detect on rent: `python scripts/detect_hardware.py` or pass `--hardware 16gb|12gb|8gb` to run scripts.

Cursor rule: [`.cursor/rules/adadae-memory-safe.mdc`](.cursor/rules/adadae-memory-safe.mdc) (dual CPU/GPU profiles)

### CPU setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Vast.ai (GPU)

**Directory layout on the instance** (datasets and code are separate repos):

```
/workspace/ITM/
  ADBench/                    # clone Minqi824/ADBench once; datasets live here
    adbench/datasets/
  project/                    # clone ITM-AdaDDAE (this repo); git pull for code updates
    configs/
    src/
    scripts/
```

Configs use `paths.adbench_root: ../ADBench/adbench/datasets` (resolved from repo root).

**One-time setup on Vast:**

```bash
cd /workspace/ITM
# ADBench already present at ./ADBench/adbench/datasets
git clone https://github.com/<you>/ITM-AdaDDAE project
cd project
bash scripts/vast_smoke.sh              # auto-detect 8/12/16 GB
```

**Code updates on Vast** (no re-clone; keep `.venv` and `results/`):

```bash
cd /workspace/ITM/project
git pull origin main
source .venv/bin/activate
python scripts/run_full_protocol.py --config configs/default_gpu.yaml --hardware rtx5070ti
```

**From your laptop** (optional rsync instead of git pull on Vast):

```bash
bash scripts/sync_to_vast.sh root@ssh1.vast.ai:PORT /workspace/ITM/project
```

Recommended offers: **RTX 5070 Ti 16 GB** `#44074990` ($0.12/hr), **RTX 5060 Ti 16 GB** `#28576024` ($0.091/hr), or **RTX 5060 8 GB** `#42076734` ($0.096/hr, use `default_gpu_8g.yaml`).

```bash
# On the instance after setup:
cd /workspace/ITM/project
bash scripts/vast_smoke.sh              # auto-detect 8/12/16 GB
# or: bash scripts/vast_smoke.sh 16gb   # force tier

# Full protocol (570 jobs, resumable via results/metrics/completed.json)
python scripts/run_full_protocol.py --config configs/default_gpu.yaml
python scripts/compare_to_ddae.py

# Before terminating — pull results back
bash scripts/sync_results_from_vast.sh root@ssh1.vast.ai:PORT /workspace/ITM/project
```

If `cu124` PyTorch fails on CUDA 13.x hosts, switch to `cu128` in `requirements-gpu.txt`.

GPU profile highlights (`default_gpu.yaml` → 16 GB `hardware_rtx5070ti.yaml`):

- VRAM soft limit 14 GB; vectorized SCS (K=64), VUS (3 draws), DTE-View
- AMP bf16, full training data, SNR-guided DANC + SSTS scoring
- 8 GB cards: use `configs/default_gpu_8g.yaml` (6.8 GB soft limit, no other changes)

## Reproduce (CPU)

```bash
# Single dataset
python scripts/run_one.py --dataset cardio --setting semi-supervised --seed 111

# Full protocol: 57 × 2 × 5 (resumable; can take multiple days on CPU)
python scripts/run_full_protocol.py --config configs/default.yaml

# Classical-only first (faster progress), then resume full run
python scripts/run_full_protocol.py --config configs/default.yaml \
  --datasets ALOI annthyroid backdoor breastw campaign cardio Cardiotocography

# Ablations (thesis)
python scripts/ablations.py --setting semi-supervised --seed 111

# Compare to published DDAE Table 1
python scripts/compare_to_ddae.py
```

Large datasets (`http`, `donors`, `census`, …) subsample **training** to 50k rows on CPU (`hardware_cpu.yaml: max_train_samples`) while keeping full evaluation; disclose this in the thesis. GPU runs use the full training split.
## Targets (DDAE, ADBench mean)

| Setting | PR-AUC | ROC-AUC |
|---------|--------|---------|
| Unsupervised | 32.77 | 74.08 |
| Semi-supervised | 61.36 | 83.17 |

## Layout

This repository **is** the project root (not nested under another folder).

```
<parent>/                     # e.g. /workspace/ITM or ~/Desktop/ITM
  ADBench/                    # separate repo — datasets only, do not commit here
    adbench/datasets/
  project/                    # this repo (ITM-AdaDDAE)
    configs/                  # default, default_gpu, hardware_* (8/12/16 GB)
    src/                      # data, models, features, train, eval, runlog, memory
    scripts/                  # run_one, run_full_protocol, vast_smoke, sync_*_vast
    results/                  # logs, metrics, ckpts, thesis tables
    thesis/                   # method notes
```

Dataset path in configs: `../ADBench/adbench/datasets` → e.g. `/workspace/ITM/ADBench/adbench/datasets`.

## Citation (baseline)

Sattarov, Schreyer, Borth. *Diffusion-Scheduled Denoising Autoencoders for Anomaly Detection in Tabular Data*. KDD 2025.
