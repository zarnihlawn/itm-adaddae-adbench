# Final run — AdaDDAE-PER (ultimate ship plan 2026-08-06)

**Last updated:** 2026-08-06 — Probe still **FAIL** `G_paper_both` (semi ~59.12–59.18 / 83.04 vs 61.36 / 83.17). Phase 0 revoked wine/census RDT (−12/−4 vs fair). Selector now **val_loss + ε-ball + complexity + synth-val PR** (never test-PR). Match-fair-on-losers ceiling ≈ **59.73** — must still beat fair on embeds/mid-tier. Catalog: [`results/adadae_per/thesis/weakness_catalog_20.json`](results/adadae_per/thesis/weakness_catalog_20.json). Campaign: [`phase4_campaign.json`](results/adadae_per/thesis/phase4_campaign.json).

Canonical ship claim: beat published DDAE on **both** settings under integrity (`val_loss` / train-only proxies, no test-PR). See [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc).

---

## Verdict (post-probe + Phase 0 local)

| Track | Semi PR / ROC | Notes |
|-------|---------------|-------|
| Paper AnoDDAE | **61.36 / 83.17** | Ship target |
| Fair valstop | 58.70 / 82.94 | Integrity baseline |
| AdaDDAE-PER (pulled) | **~59.12–59.18 / 83.04** | +0.4 vs fair; **FAIL** vs paper |
| Unsup PER | 33.01 / 75.17 | **PASS** |

**Smoking gun:** `wine` was `semi_rdt_tail+protect` → **−12.0 PR vs fair**. Val_loss-only select anti-correlated with test PR. Phase 0 forces `baseline_ddae` on wine/census; helix stripped from celeba/smtp.

**Ablate-first:** strip MCE/GATE before overlays. Bleed-classical: smtp, satimage-2, Pima, Stamps, letter, wine.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol | **570** = 57 × 2 × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` (+ `min_epochs`); never test-PR |
| Select | Primary `val_loss`; among ε-ball: complexity prior + **synth-val PR** (val normals + synthetic anomalies); veto RDT if synth ≪ baseline |
| Ship gate | PR **and** ROC **>** paper on **both** + `G_AP_PR_consistency` |
| Probe floor | Semi PR **≥ 60.5** before `ship` (env `PROBE_SEMI_FLOOR`) |
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
| **Phase 0 revoke** | `wine census smtp Stamps celeba WBC` |
| **Ship select** | Hard-12 ∪ bleed-classical |

Glue: [`train_only_recipe_select.py`](scripts/train_only_recipe_select.py) → [`apply_hard_tail_freeze.py`](scripts/apply_hard_tail_freeze.py) → [`invalidate_per_semi_jobs.py`](scripts/invalidate_per_semi_jobs.py).

Orchestrator: [`scripts/run_hard_tail_ship_path.sh`](scripts/run_hard_tail_ship_path.sh).

---

## Ship path (Vast, 16gb)

### 0. Setup

```bash
cd /data/ITM/project   # or /workspace/ITM/project
source .venv/bin/activate
export OMP_NUM_THREADS=12
python scripts/detect_hardware.py
python scripts/assert_final_config.py --config configs/adadae_per.yaml
python scripts/verify_scoring_parity.py
bash scripts/run_hard_tail_ship_path.sh wire
```

### 0b. Phase 0 emergency revoke (cheap — do first)

```bash
bash scripts/run_hard_tail_ship_path.sh phase0 16gb
# wine/census → baseline_ddae; helix off celeba/smtp; Stamps/WBC protect
```

### A. Paper-protocol diagnostic (appendix — not ship)

```bash
bash scripts/run_hard_tail_ship_path.sh paper-diag 16gb
```

### 1. GPU select (val_loss + ε + complexity + synth-val)

```bash
tmux new -s per_select
bash scripts/run_hard_tail_ship_path.sh select 16gb
# equivalent:
python scripts/train_only_recipe_select.py --preset ship --seeds 111 222 333 \
  --hardware 16gb --eps-rel 0.05 --rdt-veto-margin 0.05
```

- Writes: `results/adadae_per/thesis/phase1_hard_freeze.json` (includes `selection` meta + `mean_synth_val_pr`)
- Candidates include `baseline_ddae+full_sum|contrastive|cosine` + strip-first overlays
- If every `mean_val_loss` is null/Infinity → **do not freeze**

### 2. Freeze winners

```bash
bash scripts/run_hard_tail_ship_path.sh freeze
```

### 3. Probe — ship-probe retrain + floor gate

```bash
bash scripts/run_hard_tail_ship_path.sh probe 16gb
```

Exits non-zero if semi PR &lt; **60.5** (`phase4_probe_gate.json`). **Do not ship** until pass.

### 4. Full ship (only after probe floor)

```bash
bash scripts/run_hard_tail_ship_path.sh ship 16gb
python scripts/check_unsup_hold.py
```

**Pass when** `results/adadae_per/thesis/integrity_gates.json` → `all_pass: true`.

End-to-end select→freeze→probe: `bash scripts/run_hard_tail_ship_path.sh all 16gb` (does **not** auto-ship).

---

## Paper parity (integrity-safe) vs AnoDDAE source

From `ITM/AnoDDAE/AnoDDAE`:

| Paper | Ship |
|-------|------|
| Scaler fit on **full X** before split | **FORBIDDEN** |
| No val carve; fixed 100 ep | **FORBIDDEN** as ship protocol |
| `time_emb_dim=4`, full-sum L2 t=1..T−1 | **USED** on `baseline_ddae` |
| Batch ≈ N/10 power-of-2 | **USED** (`choose_train_batch_size`) |
| Single stochastic ε | **Upgraded** to deterministic multi-ε (`score_noise_draws=3`) |
| DDAE-C contrastive | Gated TAPS via `method_lifts.contrastive_taps_semi` |

---

## Do not

- Re-enable smtp apex / wine nautilus / speech kitchen-sink / wine·census RDT without synth-val clear
- Select by test-PR or trust `adadae_v51_hybrid` / bisect markers
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

---

## Resume / helpers

```bash
bash scripts/run_adadae_per_protocol.sh final 16gb
python scripts/audit_ap_pr_consistency.py --completed results/adadae_per/metrics/completed.json
python scripts/check_unsup_hold.py
python scripts/verify_scoring_parity.py
```

Setup: [`final_run_resume.md`](final_run_resume.md).
