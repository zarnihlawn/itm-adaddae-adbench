# Final run — AdaDDAE-PER (beat-paper freeze: classical protect + hard-tail specialists)

**Last updated:** 2026-08-05 — Primary path is **AdaDDAE-PER** with Phase0–2 beat-paper freezes (revoke A6 regressions, hard-tail recipes, FTP overrides).

Canonical ship claim: beat published DDAE on **both** unsupervised and semi-supervised under integrity protocol. See [`.cursor/rules/final-run-md.mdc`](.cursor/rules/final-run-md.mdc).

---

## What changed (beat-paper)

| Change | Detail |
|--------|--------|
| **Revoke** | smtp: apex; wine: nautilus; speech: MCE+SMC+apex kitchen-sink |
| **Protect** | High classical forced `baseline_ddae` + no A6/MCE/SMC (`protect_baseline_semi`) |
| **Hard tails** | speech/ALOI/Wilt → `semi_rdt_tail`; CIFAR/SVHN/celeba → `semi_cvnlp_ftp`; NLP → frozen + orbit |
| **A6 selective** | orbit/locus/spiral/helix on embeds only; nautilus kept on glass/Hepatitis/vertebral |
| **FTP** | per-dataset scaler/PCA/unit_norm/clip in `feature_overrides` |
| **Train** | hard-tail `min_epochs: 60`, smaller `val_fraction_semi`; donors/smtp `val_fraction_semi: 0.2` |

Configs: [`configs/adadae_per.yaml`](configs/adadae_per.yaml), [`configs/adadae_per_exceptions.yaml`](configs/adadae_per_exceptions.yaml), [`configs/adadae_per_upgrades.yaml`](configs/adadae_per_upgrades.yaml).

**Do not trust** inflated v5.1 hybrid PR with `v31_bisect_*` markers.

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol | **570** = 57 × 2 × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss` (+ `min_epochs`); never test-PR |
| Ship gate | PR **and** ROC **>** paper on **both** settings + `G_AP_PR_consistency` |
| Hardware | **`16gb`** |

| Track | Config | Results |
|-------|--------|---------|
| Fair DDAE | `configs/baselines_ddae_valstop.yaml` | `results/ddae_baseline_valstop/` |
| Paper-protocol DDAE (appendix) | `configs/ddae_paper_protocol.yaml` | `results/ddae_paper_protocol/` |
| **AdaDDAE-PER** | `configs/adadae_per.yaml` | `results/adadae_per/` |

---

## Production sequence (Vast)

```bash
cd /workspace/ITM/project
source .venv/bin/activate

python scripts/phase0_revoke_audit.py
python scripts/train_only_recipe_select.py --dry-run
# Optional GPU refine (else evidence freeze already in YAML):
# python scripts/train_only_recipe_select.py --datasets speech Wilt CIFAR10 ALOI --seeds 111 222 333 --hardware 16gb

bash scripts/run_adadae_per_protocol.sh dump_routing
bash scripts/run_adadae_per_protocol.sh invalidate
bash scripts/run_adadae_per_protocol.sh smoke 16gb
bash scripts/run_adadae_per_protocol.sh ddae 16gb
bash scripts/run_adadae_per_protocol.sh final 16gb
bash scripts/run_adadae_per_protocol.sh compare
bash scripts/run_adadae_per_protocol.sh gates
```

Or: `bash scripts/run_adadae_per_protocol.sh all 16gb`

**Pass when** `results/adadae_per/thesis/integrity_gates.json` → `all_pass: true`
(`G-I1_complete_570` + `G_AP_PR_consistency` + `G_paper_both`).

### Helpers

```bash
python scripts/audit_ap_pr_consistency.py --completed results/adadae_per/metrics/completed.json
python scripts/check_unsup_hold.py
python scripts/run_paper_protocol_diagnostic.py --datasets Wilt glass cardio --seeds 111 --hardware 16gb
python scripts/verify_scoring_parity.py
```

---

## Resume

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

## Archived

| Track | Notes |
|-------|-------|
| `adadae_champion` | Setting-only; did not clear paper-both |
| `adadae_v51_hybrid` | Bisect-inflated — quarantine |
| Tables 1–6 / v\* hybrids | Exploration only |

Setup: [`final_run_resume.md`](final_run_resume.md).
