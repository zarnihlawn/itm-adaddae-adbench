# Final run — AdaDDAE champion (one recipe)

**Last updated:** 2026-08-04 — Primary path is **one** champion 570-job track (not Tables 1–6).

This is the **canonical** command sheet for the thesis primary claim: beat published DDAE on **both** unsupervised and semi-supervised. Keep in sync when configs/protocol change (see [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc)).

---

## Integrity contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_champion.yaml` with `policy: paradigm` |
| Paradigm | Unsup → `unsup_ssts`; Semi → `champion_semi` (setting-only; **no** per-dataset routing) |
| Early stop | `val_loss` on train-carved val (`val_fraction: 0.2`); never test-PR |
| Protocol size | **570 jobs** = 57 datasets × 2 settings × seeds `{111,222,333,444,555}` |
| Shared with fair DDAE | seeds, epochs **100**, lr **0.001**, patience **30**, model **`[512,512]`**, latent **32**, `time_emb_dim` **4**, betas `0.0001→0.02` |
| Ship gate | PR **and** ROC **>** published paper means on **both** settings |
| Hard bans | VUS, `calibrated` fusion, MCE/GATE/SMC, AdaDDAE-2…6 kitchen-sink always-on |
| Hardware | Pass **`16gb`** on RTX 4060 Ti / 5070 Ti class (15–16 GB VRAM) |
| Archived | `adadae_final` / `adadae2…6` — development only, not primary |

**Frozen configs**

| Track | Config | Results |
|-------|--------|---------|
| Fair DDAE | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| **Champion (primary)** | `configs/adadae_champion.yaml` | `results/adadae_champion/` |

---

## Production sequence (Vast)

```bash
cd /workspace/ITM/project
source .venv/bin/activate
# OMP/MKL/CUDA exports active

bash scripts/run_adadae_champion_protocol.sh smoke 16gb
bash scripts/run_adadae_champion_protocol.sh ddae 16gb          # skip if already 570/570
bash scripts/run_adadae_champion_protocol.sh hard_subset 16gb   # triage before full burn
bash scripts/run_adadae_champion_protocol.sh final 16gb
bash scripts/run_adadae_champion_protocol.sh compare
bash scripts/run_adadae_champion_protocol.sh gates              # integrity + paper-both
```

Or end-to-end:

```bash
bash scripts/run_adadae_champion_protocol.sh all 16gb
```

**Pass when** `results/adadae_champion/thesis/integrity_gates.json` has `all_pass: true` including `G_paper_both`.

### Semi miss → at most 2 setting-gated patches

If paper semi fails (unsup should already clear via SSTS):

1. Patch A: light TAPS on **semi only** (`contrastive_alpha≈0.06`) inside `CHAMPION_SEMI` in `src/policy.py`
2. Patch B: mild RDT on **semi only**

Do **not** re-enable VUS / calibrated / multiview / Table-3+ modules.

---

## Resume

Jobs skip keys in `results/<run>/metrics/completed.json`. Re-run the same mode command.

```bash
python - <<'PY'
import json
from pathlib import Path
for t in ["ddae_baseline_valstop", "adadae_champion"]:
    p = Path(f"results/{t}/metrics/completed.json")
    n = len(json.loads(p.read_text()).get("completed", {})) if p.exists() else 0
    print(f"{t}: {n}/570" if p.exists() else f"{t}: MISSING")
PY
```

---

## Archived tracks (do not use for primary claims)

| Track | Notes |
|-------|-------|
| `adadae_final` … `adadae6_final` | Old Table 1–6 finals; kept for appendix / comparison |
| `adadae_v*_hybrid` | Routed hybrids — beat paper historically but **not** the primary recipe |

Setup / env / ADBench clone steps: see [`final_run_resume.md`](final_run_resume.md).
