# Final run — AdaDDAE-PER (frozen v2→v5.1 hybrid rules + Loop 2–7 improvements)

**Last updated:** 2026-08-05 — Primary path is **AdaDDAE-PER**: one runnable model that encodes the rule stack which historically beat published DDAE (v5.1 hybrid), plus integrity-preserving semi lifts (routing surgery, train horizon, selective A6).

This is the **canonical** command sheet for the thesis primary claim: beat published DDAE on **both** unsupervised and semi-supervised. Keep in sync when configs/protocol change (see [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc)).

---

## What is AdaDDAE-PER?

**One job → one recipe.** No multi-track MCE/SMC/GATE runs and no post-hoc guarded merge.

| Layer | Source | Behavior |
|-------|--------|----------|
| Base routing | v2→v4.1 + Loop 2 | [`configs/adadae_per_exceptions.yaml`](configs/adadae_per_exceptions.yaml) + [`configs/routing_rules.yaml`](configs/routing_rules.yaml) (DAMP **off**). Wilt classical; glass/vertebral/CIFAR10 → baseline |
| MCE | v5.1 − CIFAR | Modality encoder on CV/NLP/classical targets; **blocked** on semi-NLP; CIFAR10 MCE off |
| SMC | v5.1 − glass/vertebral | SNR-calibrated multiview on semi: Waveform, speech only |
| GATE | v5.1 v2 | Train-normal WTA ensemble on unsup: speech, ALOI, optdigits |
| Horizon | Loop 4 | `min_epochs: 40`, `early_stop_patience: 40`, `val_fraction_semi: 0.1` |
| Scoring | Loop 5 | baseline/nlp policies force **full-sum** L2 over `t=1..T-1` (AnoDDAE) |
| A6 | Loop 7 | Selective nautilus/apex/orbit/ridge/delta via upgrades lists |

Shell config: `configs/adadae_per.yaml` with `adadae.policy: per`.

**Do not trust** inflated v5.1 hybrid PR fields with `v31_bisect_*` markers (Wilt AP≈15 vs rewritten PR≈55). Ship gate includes `G_AP_PR_consistency`.

Historical reference (exploration only; partially bisect-inflated):
`results/adadae_v51_hybrid` ≈ unsup **37.89/78.57**, semi **62.34/85.19** vs paper **32.77/74.08** and **61.36/83.17**.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol size | **570 jobs** = 57 datasets × 2 settings × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` on train-carved val (`val_fraction` / `val_fraction_semi`); never test-PR; `min_epochs` floor |
| Ship gate | PR **and** ROC **>** published paper means on **both** settings + AP≈PR consistency |
| Routing | **Allowed** (this is the model); G-I2 no-routing does **not** apply |
| Hardware | Pass **`16gb`** on RTX 4060 Ti / 5070 Ti class (15–16 GB VRAM) |

**Frozen configs**

| Track | Config | Results |
|-------|--------|---------|
| Fair DDAE (optional compare) | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| Paper-protocol DDAE (diagnostic) | `configs/ddae_paper_protocol.yaml` | `results/ddae_paper_protocol/` |
| **AdaDDAE-PER (primary)** | `configs/adadae_per.yaml` | `results/adadae_per/` |

---

## Production sequence (Vast)

```bash
cd /workspace/ITM/project
source .venv/bin/activate

# Inspect frozen routing (optional)
bash scripts/run_adadae_per_protocol.sh dump_routing

# After Loop 2–7 config changes: invalidate stale semi jobs
python scripts/invalidate_per_semi_jobs.py

# Smoke → full 570 → compare → paper gates
bash scripts/run_adadae_per_protocol.sh smoke 16gb
bash scripts/run_adadae_per_protocol.sh ddae 16gb    # skip if already 570/570
bash scripts/run_adadae_per_protocol.sh final 16gb
bash scripts/run_adadae_per_protocol.sh compare
bash scripts/run_adadae_per_protocol.sh gates
```

Or end-to-end:

```bash
bash scripts/run_adadae_per_protocol.sh all 16gb
```

**Pass when** `results/adadae_per/thesis/integrity_gates.json` has `all_pass: true`
(`G-I1_complete_570` + `G_AP_PR_consistency` + `G_paper_both`).

### Research loop helpers

```bash
python scripts/audit_ap_pr_consistency.py --completed results/adadae_per/metrics/completed.json
python scripts/run_paper_protocol_diagnostic.py --compare-only   # after GPU run
python scripts/verify_scoring_parity.py
python scripts/train_only_recipe_select.py --dry-run
python scripts/confirm_semi_normals_only.py
python scripts/check_unsup_hold.py
```

---

## Resume

Jobs skip keys in `results/adadae_per/metrics/completed.json`. Re-run the same mode:

```bash
bash scripts/run_adadae_per_protocol.sh final 16gb
```

```bash
python - <<'PY'
import json
from pathlib import Path
for t in ["ddae_baseline_valstop", "adadae_per"]:
    p = Path(f"results/{t}/metrics/completed.json")
    n = len(json.loads(p.read_text()).get("completed", {})) if p.exists() else 0
    print(f"{t}: {n}/570" if p.exists() else f"{t}: MISSING")
PY
```

---

## Archived tracks (not primary)

| Track | Notes |
|-------|-------|
| `adadae_champion` | Setting-only paradigm; did **not** clear paper-both |
| `adadae_v51_hybrid` | Multi-track + guarded merge; **bisect-inflated** PR on some jobs — quarantine |
| `adadae_final` … `adadae6_final` | Old Table 1–6 development finals |
| `adadae_v*_hybrid` | Intermediate hybrids |

Setup / env / ADBench clone steps: see [`final_run_resume.md`](final_run_resume.md).
