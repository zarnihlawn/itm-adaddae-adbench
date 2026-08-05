# Final run — AdaDDAE-PER (20-loop audit → ablate-first ship)

**Last updated:** 2026-08-05 — Semi still **−2.18 PR** vs paper (59.18 / 83.04 vs 61.36 / 83.17); unsup holds. Match-fair-on-losses ceiling ≈ **59.82** — must beat fair on embeds. Catalog: [`results/adadae_per/thesis/weakness_catalog_20.json`](results/adadae_per/thesis/weakness_catalog_20.json).

Canonical ship claim: beat published DDAE on **both** settings under integrity (`val_loss`, no test-PR). See [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc).

---

## Verdict (20-loop)

| Track | Semi PR / ROC | Notes |
|-------|---------------|-------|
| Paper AnoDDAE | **61.36 / 83.17** | Ship target |
| Fair valstop | 58.70 / 82.94 | Integrity baseline |
| AdaDDAE-PER | **59.18 / 83.04** | +0.48 vs fair; **FAIL** vs paper |
| Unsup PER | 33.01 / 75.17 | **PASS** |

**Ablate-first:** SVHN/ALOI/celeba/CIFAR10 currently lose under kitchen-sink (MCE+GATE+multi-A6). Strip before adding overlays. Bleed-classical: smtp, satimage-2, Pima, Stamps, letter, wine.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol | **570** = 57 × 2 × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` (+ `min_epochs`); never test-PR |
| Ship gate | PR **and** ROC **>** paper on **both** + `G_AP_PR_consistency` |
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
| **Ship select** | Hard-12 ∪ bleed-classical |

Glue: [`train_only_recipe_select.py`](scripts/train_only_recipe_select.py) → [`apply_hard_tail_freeze.py`](scripts/apply_hard_tail_freeze.py) (also strips **MCE/GATE/SMC**) → [`invalidate_per_semi_jobs.py`](scripts/invalidate_per_semi_jobs.py) (`--ship-probe` / `--all-semi`).

Orchestrator: [`scripts/run_hard_tail_ship_path.sh`](scripts/run_hard_tail_ship_path.sh).

---

## Ship path (Vast, 16gb)

### 0. Setup

```bash
cd /workspace/ITM/project
source .venv/bin/activate
export OMP_NUM_THREADS=12
python scripts/detect_hardware.py
python scripts/assert_final_config.py --config configs/adadae_per.yaml
# Optional local wire (no GPU):
bash scripts/run_hard_tail_ship_path.sh wire
```

### A. Paper-protocol diagnostic (appendix — not ship)

Quantify protocol tax before attributing the whole −2.18 to recipes.

```bash
bash scripts/run_hard_tail_ship_path.sh paper-diag 16gb
# or:
python scripts/run_paper_protocol_diagnostic.py --seeds 111 222 --hardware 16gb
```

Success criteria for interpretation (`loop3_diagnostic.json`):
- If `paper_protocol` ≈ **61.36** → expect ~1.5–2.5 irreducible under integrity; need large embed lifts.
- If ≈ **fair (~58.7)** → focus 100% on recipe wins under integrity.

### 1. GPU val_loss select (strip-first candidates)

```bash
tmux new -s per_select
# pull latest first if you hit all-Infinity winners (bug: wrong DatasetSpec API + hardware string)
bash scripts/run_hard_tail_ship_path.sh select 16gb
# equivalent:
python scripts/train_only_recipe_select.py --preset ship --seeds 111 222 333 --hardware 16gb
```

- Metric: **val_loss** only (test PR logged, not used to pick).
- Candidates: plain `baseline_ddae` / `semi_cvnlp_ftp` / `semi_rdt_tail` / `champion_semi` + **at most one** of orbit|locus|helix (no MCE/GATE).
- Writes: `results/adadae_per/thesis/phase1_hard_freeze.json`
- If every `mean_val_loss` is null/Infinity → **do not freeze**; exit code ≠ 0. Delete bogus JSON and re-run after pulling the fix.
- Do **not** `apply_hard_tail_freeze` on an all-Infinity file.

### 2. Freeze winners (specialists + A6 + strip MCE/GATE)

```bash
bash scripts/run_hard_tail_ship_path.sh freeze
# or:
python scripts/apply_hard_tail_freeze.py \
  --from results/adadae_per/thesis/phase1_hard_freeze.json \
  --datasets-preset ship
python scripts/phase0_revoke_audit.py
python scripts/train_only_recipe_select.py --dry-run | head -80
```

### 3. Probe path — invalidate ship-probe (~90) + retrain

```bash
bash scripts/run_hard_tail_ship_path.sh probe 16gb
# or:
python scripts/invalidate_per_semi_jobs.py --ship-probe
python scripts/run_full_protocol.py --config configs/adadae_per.yaml --hardware 16gb \
  --datasets speech ALOI celeba SVHN CIFAR10 Wilt \
             Imdb Amazon Yelp Agnews 20newsgroups census \
             smtp satimage-2 Pima Stamps letter wine
bash scripts/run_adadae_per_protocol.sh compare
bash scripts/run_adadae_per_protocol.sh gates
```

### 4. Ship path — all semi + full 570 (recommended after probe looks good)

```bash
bash scripts/run_hard_tail_ship_path.sh ship 16gb
# or one-liner:
python scripts/invalidate_per_semi_jobs.py --all-semi \
  && bash scripts/run_adadae_per_protocol.sh final 16gb \
  && bash scripts/run_adadae_per_protocol.sh compare \
  && bash scripts/run_adadae_per_protocol.sh gates
python scripts/check_unsup_hold.py
```

**Pass when** `results/adadae_per/thesis/integrity_gates.json` → `all_pass: true`.

End-to-end (long): `bash scripts/run_hard_tail_ship_path.sh all 16gb`

---

## Do not

- Re-enable smtp apex / wine nautilus / speech kitchen-sink
- Select by test-PR or trust `adadae_v51_hybrid` / bisect markers
- Claim paper-protocol diagnostic as the ship result

---

## Sync back to laptop

```bash
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/workspace/ITM/project/results/adadae_per/ \
  /home/zarnihlawn/Desktop/ITM/project/results/adadae_per/
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/workspace/ITM/project/configs/adadae_per_*.yaml \
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
