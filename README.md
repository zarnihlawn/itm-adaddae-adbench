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
| **DTE-proxy** | DTE-inspired time/kNN score (proxy — not full ICLR 2024 DTE) |

Full showcase: [`thesis/novelty.md`](thesis/novelty.md) · Equations: [`thesis/method.md`](thesis/method.md) · Claims map: [`thesis/claims_code_map.md`](thesis/claims_code_map.md)

**Primary thesis result** = `configs/adadae_final.yaml` only (one frozen recipe, val-only early stop). Legacy hybrids / routed specialists / guarded merges are appendix history — never primary Table 1.

**Canonical final-run commands (Tables 1–5):** [`FINAL_RUN.md`](FINAL_RUN.md) — keep that file updated when protocol/configs change.

```bash
# Integrity lock before any primary run
python scripts/assert_final_config.py --config configs/adadae_final.yaml

# Phase-1 smoke (3 datasets × 2 settings × 2 seeds) — works on CPU
bash scripts/smoke_final_integrity.sh

# Full 570 on Vast GPU (fair DDAE + AdaDDAE final + Table 1 + G-I gates)
bash scripts/run_adadae_final_protocol.sh all 16gb

# AdaDDAE-2 advanced stack (after Table 1)
bash scripts/run_adadae2_protocol.sh smoke          # CPU OK
bash scripts/run_adadae2_protocol.sh all 16gb       # Vast: subset + 570 Table 2
python scripts/write_vast_run_card.py               # prints Vast commands
```

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

## AdaDDAE v2 hybrid (beat DDAE PR-AUC)

Frozen DDAE baseline: `backup/ddae_baseline_570/` (570/570, Table-1 reproduction).

**Policy routing** (`src/policy.py`): unsup → LF-DANC+MANS+SSTS; semi classical → `baselines_ddae`; semi CV/NLP → FTP+light TAPS.

```bash
# On Vast RTX 3060 12GB — smoke gates first
bash scripts/vast_adadae_v2_smoke.sh 12gb

# Production (tmux): unsup 285 + semi CV/NLP 50
bash scripts/run_adadae_v2_protocol.sh unsup   # or: semi | merge | all

# After GPU runs — merge hybrid Table 1
python scripts/merge_completed.py \
  --semi-classical backup/ddae_baseline_570/metrics/completed.json \
  --semi-cvnlp results/adadae_semi_cvnlp/metrics/completed.json \
  --unsup results/adadae_unsup_ssts/metrics/completed.json \
  --out results/adadae_v2_hybrid/metrics/completed.json \
  --copy-metrics

python scripts/compare_to_ddae.py \
  --completed results/adadae_v2_hybrid/metrics/completed.json \
  --out-dir results/adadae_v2_hybrid/thesis

python scripts/generate_hybrid_thesis.py
```

Configs: `adadae_unsup_ssts.yaml`, `adadae_semi_cvnlp.yaml`, `adadae_v2_hybrid.yaml` (routed full 570).

## AdaDDAE v3 (dataset-aware routing — beat both PR-AUC)

v3 adds per-dataset exceptions (`configs/policy_exceptions.yaml`) on top of v2 family routing.

```bash
# Local ceiling analysis (no GPU)
python scripts/oracle_policy_table.py
python scripts/build_oracle_hybrid.py --out results/adadae_v3_hybrid/metrics/completed.json
python scripts/validate_gates.py --completed results/adadae_v3_hybrid/metrics/completed.json

# On Vast — hard-dataset bisect then selective ~100-130 jobs
python scripts/v3_hard_bisect.py --hardware 12gb --epochs 100
bash scripts/run_adadae_v3_protocol.sh all

# Interim v2.1 (revert bad semi CV/NLP to backup)
python scripts/merge_completed.py \
  --semi-classical backup/ddae_baseline_570/metrics/completed.json \
  --semi-cvnlp results/adadae_semi_cvnlp/metrics/completed.json \
  --semi-cvnlp-source backup \
  --unsup results/adadae_unsup_ssts/metrics/completed.json \
  --out results/adadae_v2_1_hybrid/metrics/completed.json
```

Config: `adadae_v3_hybrid.yaml` (routed + exceptions). Oracle-best unsup PR **+4.16%** vs paper; semi still needs Vast specialist patch runs (`speech`, `ALOI`).

## Targets (DDAE, ADBench mean)

| Setting | PR-AUC | ROC-AUC |
|---------|--------|---------|
| Unsupervised | 32.77 | 74.08 |
| Semi-supervised | 61.36 | 83.17 |

## Version diagrams

Architecture, training, scoring, and protocol diagrams for every generation (DDAE → v5.1):

**[`docs/diagrams/`](docs/diagrams/)** — start at the [index](docs/diagrams/README.md) or [evolution overview](docs/diagrams/00_overview.md).

| Version | Pack |
|---------|------|
| DDAE → v3.1 | [baseline](docs/diagrams/00_ddae_baseline.md) · [v2](docs/diagrams/02_v2.md) · [v2.1](docs/diagrams/02_1_v2_1.md) · [v3](docs/diagrams/03_v3.md) · [v3.1](docs/diagrams/03_1_v31.md) |
| v4 → v5.1 | [v4](docs/diagrams/04_v4.md) · [v4.1](docs/diagrams/04_1_v41.md) · [v5](docs/diagrams/05_v5.md) · [v5.1](docs/diagrams/05_1_v51.md) |
| Cross-version | [comparison](docs/diagrams/00_comparison.md) |

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
