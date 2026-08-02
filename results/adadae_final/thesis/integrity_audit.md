# Integrity audit checklist

Generated: 2026-08-02T12:00:16.648763+00:00
Config: `/home/zarnihlawn/Desktop/ITM/project/configs/adadae_final.yaml`

## Config gate

**PASS** — primary config integrity OK

## Protocol checklist

- [x] **FTP/MCE fit on train only** — experiment carves val then fit_transform on X_fit
- [x] **DANC label_free** — adadae.danc_contamination_mode == label_free
- [x] **No test in early stop** — train.early_stop_metric == val_loss; fit rejects x_test
- [x] **No test-PR merge** — primary completed.json is not guarded hybrid
- [x] **No per-dataset specialists** — adadae.policy == static
- [x] **Val carved from train** — carve_val_from_train used in experiment.py

## Completed.json

- missing: `/home/zarnihlawn/Desktop/ITM/project/results/adadae_final/metrics/completed.json`
