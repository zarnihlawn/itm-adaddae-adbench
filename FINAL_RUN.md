# Final run (primary thesis tables)

**Last updated:** 2026-08-02 — Fresh Vast bootstrap + full Table 1→6 sequence; AdaDDAE-6 stack; Wave-6 DANC T fail-closed; protocol knobs locked to fair DDAE.

This is the **canonical** command sheet for primary Table 1–6 runs. Keep it in sync when configs, protocol scripts, assert gates, or Vast setup change (see [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc)).

## Integrity contract (do not violate)

| Rule | Value |
|------|--------|
| Recipe | One frozen YAML per table; `policy: static` |
| Early stop | `val_loss` on train-carved val (`val_fraction: 0.2`); never test-PR |
| Shared with fair DDAE | seeds, epochs, lr, patience **30**, model `[512,512]/32, `time_emb_dim` **4**, betas |
| Methods may differ | DANC / SCS / FTP / fusion / A2–A6 modules / contrastive |
| Order | **Table 1 integrity 570 first**; then Table 2 → 3 → 4 → 5 → **6** |
| Demoted | routed policy, GATE/MCE/SMC, guarded merges, oracle contamination |

Configs: `configs/adadae_final.yaml`, `adadae2_final.yaml`, `adadae3_final.yaml`, `adadae4_final.yaml`, `adadae5_final.yaml`, `adadae6_final.yaml`, `baselines_ddae_valstop.yaml`.

---

## 0. Layout and setup

Local and Vast expect ADBench as a sibling of this repo:

```
ITM/
  ADBench/adbench/datasets/
  project/          ← this repo (cwd for all commands below)
```

### Local (CPU smoke only)

```bash
cd /path/to/ITM/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Vast — fresh machine (dirs → clone → env → max hardware)

Use this on a **new** Vast instance (e.g. RTX 5070 Ti 16 GB VRAM). Prefer **≥100 GB disk** or an attached volume: a 16 GB root disk is too small for ADBench + venv + full Table 1–6 results.

**Repos are not separate git repos.** One ADBench clone ships all dataset families under `adbench/datasets/`. Primary tables use **Classical** (57 tabular sets); CV/NLP folders are unused by AdaDDAE and can be deleted to save disk.

```bash
# SSH from laptop (copy exact host/port from Vast console)
# ssh -p PORT root@sshN.vast.ai

# --- 1) Project directories ---
mkdir -p /workspace/ITM
cd /workspace/ITM
# After clones, results live under project/results/ (created by run scripts):
#   /workspace/ITM/project/results/{ddae_baseline_valstop,adadae_final,adadae2_final,...}

# --- 2) Datasets: clone ADBench once ---
git clone --depth 1 https://github.com/Minqi824/ADBench.git ADBench
# Dataset families after clone (not separate remotes):
#   ADBench/adbench/datasets/Classical/       ← required (57 tabular)
#   ADBench/adbench/datasets/CV_by_ResNet18/  ← optional / unused
#   ADBench/adbench/datasets/CV_by_ViT/       ← optional / unused (~1.3G)
#   ADBench/adbench/datasets/NLP_by_BERT/     ← optional / unused
#   ADBench/adbench/datasets/NLP_by_RoBERTa/  ← optional / unused
ls ADBench/adbench/datasets/Classical | head
# Disk saver (safe for Table 1–6): keep Classical only
rm -rf ADBench/adbench/datasets/CV_by_ResNet18 \
       ADBench/adbench/datasets/CV_by_ViT \
       ADBench/adbench/datasets/NLP_by_BERT \
       ADBench/adbench/datasets/NLP_by_RoBERTa

# --- 3) Code: clone AdaDDAE project ---
# SSH (deploy key / agent):
git clone git@github.com:zarnihlawn/itm-adaddae.git project
# HTTPS (no SSH key):
# git clone https://github.com/zarnihlawn/itm-adaddae.git project

cd /workspace/ITM/project
git checkout main
git pull origin main
mkdir -p results

# Maximize CPU for this rental (adjust to rented cores; leave a few for OS)
export OMP_NUM_THREADS=20
export MKL_NUM_THREADS=20
export OPENBLAS_NUM_THREADS=20
export NUMEXPR_NUM_THREADS=20
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

apt-get update -y && apt-get install -y tmux git
tmux new -s adadae   # long runs — Ctrl-b d to detach; tmux attach -t adadae to resume

bash scripts/setup_vast.sh 16gb    # venv + requirements-gpu.txt + cardio smoke
source .venv/bin/activate
nvidia-smi
python scripts/detect_hardware.py
# Expect suggested_profile≈hardware_rtx5070ti / 16gb on 15–16 GB cards
```

If `cu124` PyTorch fails on CUDA 13.x hosts, switch the wheel index in `requirements-gpu.txt` to `cu128` (or newer), then `pip install -r requirements-gpu.txt --force-reinstall`.

Hardware flag for all protocol scripts: **`16gb`** (default; maps to `hardware_rtx5070ti.yaml` — 14 GB VRAM soft limit, AMP bf16, vectorized scoring). Use `12gb` or `8gb` only on smaller cards.

### Vast — existing machine (code update only)

```bash
cd /workspace/ITM/project
git pull origin main
source .venv/bin/activate
python scripts/detect_hardware.py
```

### Pull results to laptop (before destroy)

```bash
# From local ITM/project
bash scripts/sync_results_from_vast.sh root@sshN.vast.ai:PORT /workspace/ITM/project
```

### One-shot full sequence (after preflight §1)

Order locked. Each step is resumable via `results/*/metrics/completed.json`.

```bash
cd /workspace/ITM/project && source .venv/bin/activate
# keep OMP/MKL_* exports from fresh setup

bash scripts/run_adadae_final_protocol.sh all 16gb   # Table 1: fair DDAE 570 + AdaDDAE 570
bash scripts/run_adadae2_protocol.sh all 16gb
bash scripts/run_adadae3_protocol.sh all 16gb
bash scripts/run_adadae4_protocol.sh all 16gb
bash scripts/run_adadae5_protocol.sh all 16gb
bash scripts/run_adadae6_protocol.sh all 16gb
```

Do **not** start Table 2–6 full 570 until Table 1 integrity 570 exists (or at least fair DDAE `ddae_baseline_valstop` for later compare).

---

## 1. Preflight (every machine, before spend)

```bash
cd /path/to/ITM/project   # or /workspace/ITM/project on Vast
source .venv/bin/activate

python scripts/assert_final_config.py --config configs/adadae_final.yaml
python scripts/assert_final_config.py --config configs/adadae2_final.yaml
python scripts/assert_final_config.py --config configs/adadae3_final.yaml
python scripts/assert_final_config.py --config configs/adadae4_final.yaml
python scripts/assert_final_config.py --config configs/adadae5_final.yaml
python scripts/assert_final_config.py --config configs/adadae6_final.yaml
python scripts/assert_final_config.py --config configs/baselines_ddae_valstop.yaml --allow-nonfinal-run-id
```

All must print `INTEGRITY OK` and `protocol_locked_to=baselines_ddae_valstop.yaml`.

Optional run card:

```bash
python scripts/write_vast_run_card.py
```

---

## 2. Table 1 — fair DDAE + AdaDDAE (`adadae_final`)

**570 jobs** = 57 datasets × `{unsupervised, semi-supervised}` × seeds `{111,222,333,444,555}`. Resumable via `results/*/metrics/completed.json`.

### One-shot on Vast (recommended)

```bash
bash scripts/run_adadae_final_protocol.sh all 16gb
```

Runs: smoke → fair DDAE 570 → AdaDDAE final 570 → compare → integrity gates G-I*.

### Step-by-step

```bash
# Local or Vast — smoke (CPU OK)
bash scripts/run_adadae_final_protocol.sh smoke
# or: bash scripts/smoke_final_integrity.sh

# Vast GPU only
bash scripts/run_adadae_final_protocol.sh ddae 16gb    # fair baseline → results/ddae_baseline_valstop/
bash scripts/run_adadae_final_protocol.sh final 16gb   # AdaDDAE → results/adadae_final/
bash scripts/run_adadae_final_protocol.sh compare
bash scripts/run_adadae_final_protocol.sh gates
```

### Reproduce table artifacts after `completed.json` exists

```bash
bash scripts/repro_final.sh 16gb
# or:
python scripts/stats_table1.py \
  --completed results/adadae_final/metrics/completed.json \
  --baseline results/ddae_baseline_valstop/metrics/completed.json \
  --out-dir results/adadae_final/thesis
```

**Artifacts:** `results/adadae_final/thesis/` (`compare_to_ddae.*`, `integrity_gates.json`, stats).

**Do not start Table 2–6 full 570 until Table 1 integrity 570 exists.**

---

## 3. Table 2 — AdaDDAE-2

```bash
bash scripts/run_adadae2_protocol.sh smoke              # CPU OK
bash scripts/run_adadae2_protocol.sh subset 16gb        # optional ladder
bash scripts/run_adadae2_protocol.sh final 16gb         # 570
bash scripts/run_adadae2_protocol.sh table2             # vs fair DDAE (+ vs Table 1 if present)

# or one-shot:
bash scripts/run_adadae2_protocol.sh all 16gb
```

Config: `configs/adadae2_final.yaml` → `results/adadae2_final/`.

---

## 4. Table 3 — AdaDDAE-3

```bash
bash scripts/run_adadae3_protocol.sh smoke
bash scripts/run_adadae3_protocol.sh subset 16gb        # optional
bash scripts/run_adadae3_protocol.sh loo 16gb           # optional
bash scripts/run_adadae3_protocol.sh tune 16gb          # optional val-only
bash scripts/run_adadae3_protocol.sh final 16gb
bash scripts/run_adadae3_protocol.sh table3

# or one-shot:
bash scripts/run_adadae3_protocol.sh all 16gb
```

Config: `configs/adadae3_final.yaml` → `results/adadae3_final/`.

---

## 5. Table 4 — AdaDDAE-4

```bash
bash scripts/run_adadae4_protocol.sh audit              # regime CSV
bash scripts/run_adadae4_protocol.sh smoke
bash scripts/run_adadae4_protocol.sh subset 16gb        # optional
bash scripts/run_adadae4_protocol.sh loo 16gb           # optional
bash scripts/run_adadae4_protocol.sh tune 16gb          # optional
bash scripts/run_adadae4_protocol.sh final 16gb
bash scripts/run_adadae4_protocol.sh table4             # + regime_breakdown.json

# or one-shot:
bash scripts/run_adadae4_protocol.sh all 16gb
```

Config: `configs/adadae4_final.yaml` → `results/adadae4_final/`.

---

## 6. Table 5 — AdaDDAE-5 (information-geometric)

After fair DDAE (+ ideally Table 1). Stretch metric ladder: G5-1 vs fair DDAE → G5-2/G5-3 absolute; unsup PR ≥80 macro is a disclosed moonshot.

**Wave-6:** DANC `_resolve_T_from_snr` fail-closes to `T_max` (semi no longer stuck at T=5); FIGARO T★ is a lower bound; CONFAL fits on EVT-transformed scores; `use_elbo_s: false` by default; `auto_regime_gates` may enable GEODE/ORBIS on high-d. **Re-run Table 1 smoke/570 after DANC T fix** (realized T changes; protocol knobs unchanged).

Ablation ladder steps (wired): `adadae5_core` → `figaro_dsm` → `mahala_dte` → `lexicon_fuse` → `full_adadae5`. LOO uses `LEAVE_ONE_OUT_A5` (not A2).

```bash
bash scripts/run_adadae5_protocol.sh smoke
bash scripts/run_adadae5_protocol.sh smoke_hard   # thyroid/letter/speech, PR floor 0.01 (speech rare)
bash scripts/run_adadae5_protocol.sh subset 16gb        # optional ladder (fails loud if steps missing)
bash scripts/run_adadae5_protocol.sh loo 16gb           # optional A5 LOO
bash scripts/run_adadae5_protocol.sh tune 16gb          # optional val-only
bash scripts/run_adadae5_protocol.sh final 16gb
bash scripts/run_adadae5_protocol.sh table5

# or one-shot:
bash scripts/run_adadae5_protocol.sh all 16gb
```

Config: `configs/adadae5_final.yaml` → `results/adadae5_final/`.

---

## 6b. Table 6 — AdaDDAE-6 (ADBench regime stack)

After Table 5. Catalog: [`thesis/adbench_improve_catalog.md`](thesis/adbench_improve_catalog.md). Modules: HELIX, DELTA, APEX, NAUTILUS, TORQUE, ORBIT, KALE, RIDGE, LOCUS, SPIRAL on A5 core. Fusion default: `kale`.

```bash
bash scripts/run_adadae6_protocol.sh smoke
bash scripts/run_adadae6_protocol.sh smoke_hard
bash scripts/run_adadae6_protocol.sh subset 16gb
bash scripts/run_adadae6_protocol.sh loo 16gb
bash scripts/run_adadae6_protocol.sh final 16gb
bash scripts/run_adadae6_protocol.sh table6

# or one-shot:
bash scripts/run_adadae6_protocol.sh all 16gb
```

Config: `configs/adadae6_final.yaml` → `results/adadae6_final/`.

---

## 7. Modes cheat sheet

| Script | Modes |
|--------|--------|
| `run_adadae_final_protocol.sh` | `smoke` \| `ddae` \| `final` \| `compare` \| `gates` \| `all` |
| `run_adadae2_protocol.sh` | `smoke` \| `subset` \| `final` \| `table2` \| `all` |
| `run_adadae3_protocol.sh` | `smoke` \| `subset` \| `loo` \| `tune` \| `final` \| `table3` \| `all` |
| `run_adadae4_protocol.sh` | `audit` \| `smoke` \| `subset` \| `loo` \| `tune` \| `final` \| `table4` \| `all` |
| `run_adadae5_protocol.sh` | `smoke` \| `smoke_hard` \| `subset` \| `loo` \| `tune` \| `final` \| `table5` \| `all` |
| `run_adadae6_protocol.sh` | `smoke` \| `smoke_hard` \| `subset` \| `loo` \| `final` \| `table6` \| `all` |

Second arg (optional): hardware tier `16gb` \| `12gb` \| `8gb`.

---

## 8. Key paths

| What | Path |
|------|------|
| Fair DDAE completed | `results/ddae_baseline_valstop/metrics/completed.json` |
| Table 1 completed | `results/adadae_final/metrics/completed.json` |
| Table 2–6 completed | `results/adadae{2,3,4,5,6}_final/metrics/completed.json` |
| Thesis outputs | `results/adadae*_final/thesis/` |
| Regime audit (A4) | `results/thesis/adbench_regimes.csv` |
| Claims map | `thesis/claims_code_map.md` |
| Method / protocol | `thesis/method.md` |

---

## 9. Out of scope for primary tables

Do **not** use these for Table 1–6 claims:

- `run_adadae_v*_protocol.sh` (v2–v5.1 hybrids)
- `policy: routed`, `policy_exceptions.yaml`, `merge_v5_guarded.py`
- Smoke YAMLs (`*_final_smoke.yaml`) — debug only
- Comparing only to published paper means without `ddae_baseline_valstop` (paper means are secondary reference via `compare_to_ddae.py`)

---

## Maintainer note

When you change any of: `configs/*_final.yaml`, `baselines_ddae_valstop.yaml`, `scripts/run_adadae*_protocol.sh`, `scripts/assert_final_config.py`, `scripts/repro_final.sh`, `scripts/smoke_final_integrity.*`, Vast setup/sync scripts, or the integrity contract — **update this file in the same change** (commands, paths, knobs, last-updated date).
