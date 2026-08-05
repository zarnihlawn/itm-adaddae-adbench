# Final run — AdaDDAE-PER (frozen v2→v5.1 hybrid rules)

**Last updated:** 2026-08-05 — Primary path is **AdaDDAE-PER**: one runnable model that encodes the rule stack which historically beat published DDAE (v5.1 hybrid).

This is the **canonical** command sheet for the thesis primary claim: beat published DDAE on **both** unsupervised and semi-supervised. Keep in sync when configs/protocol change (see [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc)).

---

## What is AdaDDAE-PER?

**One job → one recipe.** No multi-track MCE/SMC/GATE runs and no post-hoc guarded merge.

| Layer | Source | Behavior |
|-------|--------|----------|
| Base routing | v2→v4.1 | `configs/adadae_per_exceptions.yaml` + `configs/routing_rules.yaml` (DAMP **off**) |
| MCE | v5.1 | Modality encoder on CV/NLP/classical targets; **blocked** on semi-NLP |
| SMC | v5.1 | SNR-calibrated multiview fusion on semi tails: glass, vertebral, Waveform, speech |
| GATE | v5.1 v2 | Train-normal WTA ensemble on unsup: speech, ALOI, optdigits |

Shell config: `configs/adadae_per.yaml` with `adadae.policy: per`.

Historical reference numbers (prior multi-track hybrid, for calibration):
`results/adadae_v51_hybrid` ≈ unsup **37.89/78.57**, semi **62.34/85.19** vs paper **32.77/74.08** and **61.36/83.17**.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol size | **570 jobs** = 57 datasets × 2 settings × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` on train-carved val (`val_fraction: 0.2`); never test-PR |
| Ship gate | PR **and** ROC **>** published paper means on **both** settings |
| Routing | **Allowed** (this is the model); G-I2 no-routing does **not** apply |
| Hardware | Pass **`16gb`** on RTX 4060 Ti / 5070 Ti class (15–16 GB VRAM) |

**Frozen configs**

| Track | Config | Results |
|-------|--------|---------|
| Fair DDAE (optional compare) | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| **AdaDDAE-PER (primary)** | `configs/adadae_per.yaml` | `results/adadae_per/` |

---

## Production sequence (Vast)

```bash
cd /workspace/ITM/project
source .venv/bin/activate

# Inspect frozen routing (optional)
bash scripts/run_adadae_per_protocol.sh dump_routing

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

**Pass when** `results/adadae_per/thesis/integrity_gates.json` has `all_pass: true` (`G-I1_complete_570` + `G_paper_both`).

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
| `adadae_v51_hybrid` | Multi-track + guarded merge that PER freezes into one pass |
| `adadae_final` … `adadae6_final` | Old Table 1–6 development finals |
| `adadae_v*_hybrid` | Intermediate hybrids |

Setup / env / ADBench clone steps: see [`final_run_resume.md`](final_run_resume.md).
