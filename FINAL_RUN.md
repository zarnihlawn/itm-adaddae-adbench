# Final run — production Vast (Tables 1–6)

**Last updated:** 2026-08-02 — Production-only sheet: full 57-dataset × 2 settings × 5 seeds on Vast GPU. No local CPU, no smoke/subset/LOO/tune ladders.

This is the **canonical** command sheet for thesis primary tables. Keep it in sync when configs, protocol scripts, assert gates, or Vast setup change (see [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc)).

---

## Integrity contract (do not violate)

| Rule | Value |
|------|--------|
| Recipe | One frozen YAML per table; `policy: static` |
| Early stop | `val_loss` on train-carved val (`val_fraction: 0.2`); never test-PR |
| Protocol size | **570 jobs** per track = **57 datasets** × `{unsupervised, semi-supervised}` × seeds `{111,222,333,444,555}` |
| 57 datasets | **47 Classical + 5 CV (ResNet18) + 5 NLP (BERT)** — same as DDAE / `thesis/method.md` |
| Shared with fair DDAE | seeds, epochs **100**, lr **0.001**, patience **30**, model **`[512,512]`**, latent **32**, `time_emb_dim` **4**, betas `0.0001→0.02`, linear scheduler, `num_timesteps` **50** |
| Methods may differ | DANC / SCS / FTP / fusion / A2–A6 modules / contrastive |
| Order | **Fair DDAE + Table 1 first**; then Table **2 → 3 → 4 → 5 → 6** |
| Hardware | Always pass **`16gb`** on RTX 5070 Ti / 5060 Ti class (15–16 GB VRAM) |
| Demoted | routed policy, GATE/MCE/SMC, guarded merges, oracle contamination, `run_adadae_v*_protocol.sh` |

**Frozen configs**

| Track | Config | `run_id` / results |
|-------|--------|-------------------|
| Fair DDAE | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| Table 1 | `configs/adadae_final.yaml` | `results/adadae_final/` |
| Table 2 | `configs/adadae2_final.yaml` | `results/adadae2_final/` |
| Table 3 | `configs/adadae3_final.yaml` | `results/adadae3_final/` |
| Table 4 | `configs/adadae4_final.yaml` | `results/adadae4_final/` |
| Table 5 | `configs/adadae5_final.yaml` | `results/adadae5_final/` |
| Table 6 | `configs/adadae6_final.yaml` | `results/adadae6_final/` |

All final YAMLs set `hardware: hardware_rtx5070ti.yaml` and `paths.adbench_root: ../ADBench/adbench/datasets`.

---

## 1. Machine requirements

| Resource | Minimum for full Tables 1–6 |
|----------|-----------------------------|
| GPU | CUDA, **≥15 GB VRAM** (RTX 5070 Ti / 5060 Ti) → profile **`16gb`** |
| Disk | **≥100 GB** free (or Vast volume). **16 GB root is not enough** |
| RAM | ≥32 GB recommended (`rss_soft_limit_mb: 28672` in hardware profile) |
| Continuity | `tmux` (or equivalent) — runs take days; jobs resume via `completed.json` |

**Hardware profile (`configs/hardware_rtx5070ti.yaml`) — full feature set**

| Knob | Value |
|------|--------|
| `device` | `cuda` |
| `vram_soft_limit_mb` | `14000` |
| `rss_soft_limit_mb` | `28672` |
| `train_batch_size_max` | `4096` |
| `score_batch_size_max` | `8192` |
| `use_amp` / `amp_dtype` | `true` / `bfloat16` |
| `pin_memory` | `true` |
| `cudnn_benchmark` | `true` |
| `vectorized_scoring` | `true` |
| `max_train_samples` | `0` (full data) |
| `dataloader_num_workers` | `4` |
| `num_threads` | `8` (override further with `OMP_*` below) |

---

## 2. Layout (fixed)

```
/workspace/ITM/
  ADBench/
    adbench/datasets/
      Classical/          ← required (47)
      CV_by_ResNet18/     ← required (5 families)
      NLP_by_BERT/        ← required (5 families)
      CV_by_ViT/          ← unused by this repo (optional delete)
      NLP_by_RoBERTa/     ← unused by this repo (optional delete)
  project/                ← this repo; cwd for every command below
    configs/
    src/
    scripts/
    results/              ← written by protocols
```

---

## 3. Fresh Vast — directories, clones, env, max resources

SSH from the Vast console (host/port are instance-specific):

```bash
ssh -p PORT root@sshN.vast.ai
```

### 3.1 Create directories

```bash
mkdir -p /workspace/ITM
cd /workspace/ITM
```

### 3.2 Clone datasets (one repo — all 57 families)

```bash
git clone --depth 1 https://github.com/Minqi824/ADBench.git ADBench

# Verify required families (do NOT delete these three)
ls ADBench/adbench/datasets/Classical | wc -l          # expect 47 .npz (+ maybe extras)
ls ADBench/adbench/datasets/CV_by_ResNet18 | head
ls ADBench/adbench/datasets/NLP_by_BERT | head

# Optional ~1.5G saver — alternate embeds unused by AdaDDAE loaders
rm -rf ADBench/adbench/datasets/CV_by_ViT \
       ADBench/adbench/datasets/NLP_by_RoBERTa
```

### 3.3 Clone project

```bash
# Prefer SSH if a deploy key is on the instance:
git clone git@github.com:zarnihlawn/itm-adaddae.git project
# Else HTTPS:
# git clone https://github.com/zarnihlawn/itm-adaddae.git project

cd /workspace/ITM/project
git checkout main
git pull origin main
mkdir -p results
```

### 3.4 Maximize CPU / GPU for this rental

Adjust thread counts to rented cores (example: 24 visible → use 20):

```bash
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

apt-get update -y && apt-get install -y tmux git rsync \
  python3 python3-venv python3-pip python3-full python3-dev
tmux new -s adadae
# Detach: Ctrl-b d | Reattach: tmux attach -t adadae
```

Put the same `export` lines in `~/.bashrc` or re-export after every new shell / `tmux` pane.

### 3.5 GPU Python env

Vast images often ship a broken/minimal Python (`ensurepip` fails, or `_posixsubprocess` missing). **Wipe any half-made `.venv` first**, fix system packages, then recreate.

```bash
cd /workspace/ITM/project

# Leave a broken activate if you already sourced it
deactivate 2>/dev/null || true
unset PYTHONHOME PYTHONPATH
rm -rf .venv

# System Python must import stdlib C extensions before venv
python3 -c "import subprocess, _posixsubprocess; print(python3_ok := 'ok')"

python3 -m venv .venv
source .venv/bin/activate
python -c "import subprocess; print('venv_ok')"
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

If `python3 -m venv` still fails on `ensurepip`, bootstrap pip manually:

```bash
rm -rf .venv
python3 -m venv --without-pip .venv
source .venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python /tmp/get-pip.py
pip install --upgrade pip
pip install -r requirements-gpu.txt
```

If **system** `python3 -c "import _posixsubprocess"` fails, reinstall the interpreter (do not keep using the broken `.venv`):

```bash
apt-get install --reinstall -y python3 python3.12-minimal libpython3.12-stdlib
# then recreate .venv from the top of this section
```

If install/import fails on **CUDA 13.x** hosts, edit `requirements-gpu.txt`: change `--index-url` from `cu124` to `cu128` (or newer matching the driver), then:

```bash
pip install -r requirements-gpu.txt --force-reinstall
```

Confirm:

```bash
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA required for final run"
print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0))
print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
print("bf16", torch.cuda.is_bf16_supported())
PY
python scripts/detect_hardware.py
# Expect suggested_profile ≈ hardware_rtx5070ti / 16gb
```

### 3.6 Code update on an already-cloned instance

```bash
cd /workspace/ITM/project
git pull origin main
source .venv/bin/activate
# re-export OMP/MKL/CUDA_* if needed
```

---

## 4. Preflight (before burning GPU hours)

```bash
cd /workspace/ITM/project
source .venv/bin/activate

python scripts/assert_final_config.py --config configs/adadae_final.yaml
python scripts/assert_final_config.py --config configs/adadae2_final.yaml
python scripts/assert_final_config.py --config configs/adadae3_final.yaml
python scripts/assert_final_config.py --config configs/adadae4_final.yaml
python scripts/assert_final_config.py --config configs/adadae5_final.yaml
python scripts/assert_final_config.py --config configs/adadae6_final.yaml
python scripts/assert_final_config.py --config configs/baselines_ddae_valstop.yaml --allow-nonfinal-run-id

python scripts/write_vast_run_card.py
```

Every assert must print **`INTEGRITY OK`** and `protocol_locked_to=baselines_ddae_valstop.yaml`.

Dataset path check:

```bash
test -d ../ADBench/adbench/datasets/Classical
test -d ../ADBench/adbench/datasets/CV_by_ResNet18
test -d ../ADBench/adbench/datasets/NLP_by_BERT
df -h /workspace
```

---

## 5. Master production sequence

**Do not** use protocol mode `all` / `smoke` / `subset` / `loo` / `tune` / `smoke_hard` for the primary thesis burn — those are development ladders. Production path below is **full 570 + table artifacts only**.

```bash
cd /workspace/ITM/project
source .venv/bin/activate
# OMP/MKL/CUDA exports from §3.4 must be active

# ----- Table 1: fair DDAE + AdaDDAE -----
bash scripts/run_adadae_final_protocol.sh ddae 16gb
bash scripts/run_adadae_final_protocol.sh final 16gb
bash scripts/run_adadae_final_protocol.sh compare
bash scripts/run_adadae_final_protocol.sh gates

# ----- Table 2 -----
bash scripts/run_adadae2_protocol.sh final 16gb
bash scripts/run_adadae2_protocol.sh table2

# ----- Table 3 -----
bash scripts/run_adadae3_protocol.sh final 16gb
bash scripts/run_adadae3_protocol.sh table3

# ----- Table 4 (audit builds regime CSV used by table4) -----
bash scripts/run_adadae4_protocol.sh audit
bash scripts/run_adadae4_protocol.sh final 16gb
bash scripts/run_adadae4_protocol.sh table4

# ----- Table 5 -----
bash scripts/run_adadae5_protocol.sh final 16gb
bash scripts/run_adadae5_protocol.sh table5

# ----- Table 6 -----
bash scripts/run_adadae6_protocol.sh final 16gb
bash scripts/run_adadae6_protocol.sh table6
```

**Gate:** do not start Table 2–6 `final` until `results/ddae_baseline_valstop/metrics/completed.json` and `results/adadae_final/metrics/completed.json` exist (Table 1 compare/gates done).

### Resume after disconnect / kill

Re-run the same `… final 16gb` command. `run_full_protocol.py` skips keys already in `results/<run>/metrics/completed.json`.

### Completion check (expect 570 per track)

```bash
python - <<'PY'
import json
from pathlib import Path
tracks = [
  "ddae_baseline_valstop",
  "adadae_final",
  "adadae2_final",
  "adadae3_final",
  "adadae4_final",
  "adadae5_final",
  "adadae6_final",
]
for t in tracks:
    p = Path(f"results/{t}/metrics/completed.json")
    if not p.exists():
        print(f"{t}: MISSING")
        continue
    n = len(json.loads(p.read_text()).get("completed", {}))
    print(f"{t}: {n}/570")
PY
```

---

## 6. Per-table feature detail (what each frozen YAML enables)

Shared on **every** final track (locked by `assert_final_config.py`):

- `policy: static`
- `early_stop_metric: val_loss`, `early_stop_patience: 30`, `val_fraction: 0.2`
- `epochs: 100`, `lr: 0.001`, `eval_every: 10`
- `model.hidden_dims: [512, 512]`, `latent_dim: 32`
- `diffusion`: `num_timesteps: 50`, `beta_start/end: 0.0001/0.02`, `time_emb_dim: 4`
- `use_mce: false`, `use_gate: false`
- Features: `scaler: auto`, PCA threshold 128, clip outliers σ=5

### Fair DDAE — `baselines_ddae_valstop.yaml`

| Feature | On |
|---------|----|
| Contrastive / DANC / SCS / FTP / multiview / fusion extras | **Off** (plain DDAE under same val-stop protocol) |

### Table 1 — `adadae_final.yaml` — fusion `calibrated`

| Feature | On |
|---------|----|
| DANC (label-free) | yes |
| SCS (`snr_weighted`, stratified, max T 64) | yes |
| FTP | yes |
| Multiview + uncertainty (3 draws) + DTE-View | yes |
| Rejection training (RDT) | yes |
| Contrastive + TAPS pairing + hard negatives + adaptive α | yes |
| Fusion weights | recon/latent/residual/uncertainty/diffusion_time |

### Table 2 — `adadae2_final.yaml` — fusion `calix`

Adds / changes vs Table 1 core: **CHRONOS**, **GEODE**, **AETHER**, **NEXUS**; uncertainty view **off**; DTE-View + RDT on.

### Table 3 — `adadae3_final.yaml` — fusion `calix`

Adds: **HELIOS**, **KAIROS**, **ORBIS**, **PLEXUS**, **NEXUS_V2**, **RDT_V2**, **EPOCHE**; NEXUS off; STRATA/PHASOR/ARGOS/ATLAS/HYDRA/FLUX/SCRIBE off.

### Table 4 — `adadae4_final.yaml` — fusion `quell`

Adds: **OMNI**, **NANO**, **TORRENT**, **PRISM**, **POLIS**, **SIEVE**, **NEEDLE**, **SPARSE_VIEW**, **ROBUST**, **QUELL**, plus Table-3 stack (GEODE/AETHER/HELIOS/KAIROS/ORBIS/PLEXUS/NEXUS_V2/RDT_V2/EPOCHE). MIRAGE/AEGIS/FLUX/ATLAS off.

Regime audit artifact: `results/thesis/adbench_regimes.csv` (from `audit`).

### Table 5 — `adadae5_final.yaml` — fusion `lexicon` (info-geometry)

A2–A4 regime modules default **off** (`auto_regime_gates` may enable GEODE/ORBIS on high-d). **On:** FIGARO, DSM+, MAHALA, full DTE, LEXICON, PURA, EVT tail, CONFAL, SPECTRA, SINKHORN, IB latent, curriculum SNR, vMF-z. **Off:** ELBO-S. Wave-6: DANC T fail-closes to `T_max`; FIGARO T★ lower bound; CONFAL on EVT scores.

### Table 6 — `adadae6_final.yaml` — fusion `kale`

A5 core **on** + ADBench 10-loop stack **on:** HELIX, DELTA, APEX, NAUTILUS, TORQUE, ORBIT, KALE, RIDGE, LOCUS, SPIRAL. Catalog: [`thesis/adbench_improve_catalog.md`](thesis/adbench_improve_catalog.md).

---

## 7. Artifacts (must exist after each track)

| Track | `completed.json` | Thesis outputs |
|-------|------------------|----------------|
| Fair DDAE | `results/ddae_baseline_valstop/metrics/completed.json` | — |
| Table 1 | `results/adadae_final/metrics/completed.json` | `results/adadae_final/thesis/` (`compare_to_ddae.*`, `integrity_gates.json`, stats) |
| Table 2 | `results/adadae2_final/metrics/completed.json` | `results/adadae2_final/thesis/` |
| Table 3 | `results/adadae3_final/metrics/completed.json` | `results/adadae3_final/thesis/` |
| Table 4 | `results/adadae4_final/metrics/completed.json` | `results/adadae4_final/thesis/` (+ `regime_breakdown.json`) |
| Table 5 | `results/adadae5_final/metrics/completed.json` | `results/adadae5_final/thesis/` |
| Table 6 | `results/adadae6_final/metrics/completed.json` | `results/adadae6_final/thesis/` |

Claims / method references: [`thesis/claims_code_map.md`](thesis/claims_code_map.md), [`thesis/method.md`](thesis/method.md), [`thesis/novelty.md`](thesis/novelty.md).

---

## 8. Pull results to laptop (before destroy)

On the **laptop**, from local `ITM/project`:

```bash
bash scripts/sync_results_from_vast.sh root@sshN.vast.ai:PORT /workspace/ITM/project
```

Equivalent:

```bash
rsync -avz --progress -e 'ssh -p PORT' \
  root@sshN.vast.ai:/workspace/ITM/project/results/ \
  ./results/
```

---

## 9. Forbidden for primary tables

Do **not** use for Table 1–6 claims:

- Local CPU / `requirements.txt` path
- Protocol modes: `smoke`, `smoke_hard`, `subset`, `loo`, `tune`, or umbrella `all` (those embed development ladders)
- Smoke YAMLs (`*_final_smoke.yaml`, `*_smoke_hard.yaml`) as the reported recipe
- `run_adadae_v*_protocol.sh` (v2–v5.1 hybrids)
- `policy: routed`, `policy_exceptions.yaml`, `merge_v5_guarded.py`
- Comparing only to published paper means without `ddae_baseline_valstop` (paper means are secondary via `compare_to_ddae.py`)
- Deleting `Classical/`, `CV_by_ResNet18/`, or `NLP_by_BERT/` (breaks the 57-dataset protocol)

---

## Maintainer note

When you change any of: `configs/*_final.yaml`, `baselines_ddae_valstop.yaml`, `hardware_rtx5070ti.yaml`, `scripts/run_adadae*_protocol.sh`, `scripts/assert_final_config.py`, Vast setup/sync scripts, dataset layout, or the integrity contract — **update this file in the same change** (commands, paths, feature tables, last-updated date).
