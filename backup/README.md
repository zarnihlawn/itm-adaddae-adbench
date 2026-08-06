# Baseline results backup

## DDAE-PAR (full project archive)

**DDAE-PAR** = Policy-Adaptive Router (internal run `adadae_per`, last-shot 2026-08-06).

Full-tree achievement snapshot (no `.venv`):

[`../../archive/DDAE-PAR_2026-08-06/`](../../archive/DDAE-PAR_2026-08-06/) · [`ACHIEVED.md`](../../archive/DDAE-PAR_2026-08-06/ACHIEVED.md)

| Setting | PR / ROC | vs paper DDAE | vs fair DDAE |
|---------|----------|---------------|--------------|
| Unsupervised | 33.01 / 75.17 | +0.24 / +1.09 | +7.20 / +5.62 |
| Semi-supervised | 59.01 / 83.08 | −2.35 / −0.09 | +0.31 / +0.14 |

---

Frozen copy of the **DDAE-faithful 570-job run** (`configs/baselines_ddae.yaml`, RTX 3060 12GB, Vast #44704345).

## Contents

- `ddae_baseline_570/metrics/` — `completed.json` + per-job JSON (570 jobs)
- `ddae_baseline_570/thesis/` — `compare_to_ddae.csv`, per-dataset tables
- `ddae_baseline_570/logs/` — run logs (if any)

## Published comparison (Table 1)

| Setting | PR-AUC | ROC-AUC | vs DDAE PR | vs DDAE ROC |
|---------|--------|---------|------------|-------------|
| Unsupervised | 32.63% | 75.81% | −0.14% | +1.73% |
| Semi-supervised | 60.75% | 84.29% | −0.61% | +1.12% |

DDAE targets: unsup PR 32.77 / ROC 74.08; semi PR 61.36 / ROC 83.17.

## Re-run comparison from this backup

```bash
python3 scripts/compare_to_ddae.py \
  --completed backup/ddae_baseline_570/metrics/completed.json
```

## New experiments

Active results go to `results/` (empty after backup). Use a dedicated subfolder for AdaDDAE v2, e.g. `results/adadae_v2/` in config `paths.results_dir`.
