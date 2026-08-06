# Final run — AdaDDAE-PER (Phase0 lock + paper gap plan)

**Last updated:** 2026-08-06 — **Phase0 lock shipped in configs/code.** Select→freeze had undone wine/census → `champion_semi` / `semi_rdt_tail` (live metrics wine −4.26, census −4.02). Both are now hard-locked to `baseline_ddae`; Agnews TAPS stripped. Macros still **pre-retrain** (59.09 / 82.98 vs paper 61.36 / 83.17). **Next on Vast:** `bash scripts/run_phase0_lock_retrain.sh top3 16gb`. Catalog: [`results/adadae_per/thesis/weakness_catalog_20.json`](results/adadae_per/thesis/weakness_catalog_20.json).

Canonical ship claim: beat published DDAE on **both** settings under integrity (`val_loss` / train-only proxies, no test-PR). See [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc).

---

## Verdict (post-probe; pre Phase0-lock retrain)

| Track | Semi PR / ROC | Notes |
|-------|---------------|-------|
| Paper AnoDDAE | **61.36 / 83.17** | Ship target |
| Fair valstop | 58.70 / 82.94 | Integrity baseline |
| AdaDDAE-PER (metrics on disk) | **59.09 / 82.98** | +0.39 vs fair; **FAIL** vs paper |
| Unsup PER | 33.01 / 75.17 | **PASS** |

**Smoking gun (updated):** Phase0 emergency revoke was **undone** by GPU select→freeze. Wine ran `champion_semi+protect` (FTP); census ran `semi_rdt_tail` again. **PHASE0_LOCKS** in [`apply_hard_tail_freeze.py`](scripts/apply_hard_tail_freeze.py) + [`train_only_recipe_select.py`](scripts/train_only_recipe_select.py) force `baseline_ddae` forever. Agnews TAPS removed from [`adadae_per_upgrades.yaml`](configs/adadae_per_upgrades.yaml).

**Ceiling:** match-fair-on-losers ≈ **59.72** — still **−1.64** vs paper → must beat fair on mid-tier after stopping the bleed.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol | **570** = 57 × 2 × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` (+ `min_epochs`); never test-PR |
| Select | Primary `val_loss`; ε-ball + complexity + synth-val; **PHASE0_LOCKS** wine/census |
| Ship gate | PR **and** ROC **>** paper on **both** + `G_AP_PR_consistency` |
| Probe floor | Semi PR **≥ 60.5** before `ship` |
| Hardware | **`16gb`** |

| Track | Config | Results |
|-------|--------|---------|
| Fair DDAE | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| Paper-protocol (appendix) | `configs/ddae_paper_protocol.yaml` | `results/ddae_paper_protocol/` |
| **AdaDDAE-PER** | `configs/adadae_per.yaml` | `results/adadae_per/` |

---

## Sets

| Set | Datasets |
|-----|----------|
| **Hard-12** | `speech ALOI celeba SVHN CIFAR10 Wilt Imdb Amazon Yelp Agnews 20newsgroups census` |
| **Bleed-CV** | `SVHN ALOI celeba CIFAR10 speech` |
| **Bleed-classical** | `smtp satimage-2 Pima Stamps letter wine` |
| **Phase 0 lock** | `wine census` → always `baseline_ddae` |
| **Protect** | classical list + **Pima letter** (new) |
| **Cal-fuse expand** | + `backdoor Hepatitis cover` |

---

## Ship path (Vast, 16gb)

### 0. Setup

```bash
cd /data/ITM/project   # or /workspace/ITM/project
source .venv/bin/activate
export OMP_NUM_THREADS=12
python scripts/detect_hardware.py
python scripts/assert_final_config.py --config configs/adadae_per.yaml
bash scripts/run_phase0_lock_retrain.sh audit
```

### 1. Phase0-lock retrain (do first — configs already locked)

```bash
bash scripts/run_phase0_lock_retrain.sh top3 16gb
# wine census Agnews invalidate + retrain + compare
```

Expect: wine/census near fair; Agnews Δ better than −3.2; semi macro ~59.3–59.6.

### 2. Protect residuals

```bash
bash scripts/run_phase0_lock_retrain.sh residuals 16gb
# Stamps smtp Pima letter WBC
```

### 3. Mid-tier lifts → probe

```bash
bash scripts/run_phase0_lock_retrain.sh midtier 16gb
# backdoor Hepatitis cover + annthyroid campaign yeast MVTec-AD cardio
# then probe floor 60.5
```

### 4. Full ship (only after probe PASS)

```bash
bash scripts/run_hard_tail_ship_path.sh ship 16gb
python scripts/check_unsup_hold.py
```

**Pass when** `results/adadae_per/thesis/integrity_gates.json` → `all_pass: true`.

---

## Legacy select path (still valid; locks enforced)

```bash
bash scripts/run_hard_tail_ship_path.sh select 16gb
bash scripts/run_hard_tail_ship_path.sh freeze   # Phase0 locks override wine/census
bash scripts/run_hard_tail_ship_path.sh probe 16gb
```

---

## Paper parity (integrity-safe) vs AnoDDAE source

| Paper | Ship |
|-------|------|
| Scaler fit on **full X** before split | **FORBIDDEN** |
| No val carve; fixed 100 ep | **FORBIDDEN** as ship protocol |
| `time_emb_dim=4`, full-sum L2 t=1..T−1 | **USED** on `baseline_ddae` |
| Batch ≈ N/10 power-of-2 | **USED** |
| Single stochastic ε | **Upgraded** to multi-ε (`score_noise_draws=3`) |

---

## Do not

- Re-enable wine·census RDT / wine champion_semi without clearing PHASE0_LOCKS
- Select by test-PR or trust bisect-inflated hybrid PR
- Full-ship when probe semi PR &lt; 60.5
- Claim paper-protocol diagnostic as the ship result

---

## Sync back to laptop

```bash
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/data/ITM/project/results/adadae_per/ \
  /home/zarnihlawn/Desktop/ITM/project/results/adadae_per/
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/data/ITM/project/configs/adadae_per_*.yaml \
  /home/zarnihlawn/Desktop/ITM/project/configs/
```

Setup: [`final_run_resume.md`](final_run_resume.md).
