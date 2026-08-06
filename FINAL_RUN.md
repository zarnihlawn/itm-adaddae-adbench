# Final run — AdaDDAE-PER (last-shot adaptive, one Vast command)

**Achievement name:** **DDAE-PAR** (Policy-Adaptive Router). Full-project archive: [`../archive/DDAE-PAR_2026-08-06/`](../archive/DDAE-PAR_2026-08-06/) · [`ACHIEVED.md`](../archive/DDAE-PAR_2026-08-06/ACHIEVED.md).

**Last updated:** 2026-08-06 — **Last-shot configs frozen.** Full-57 adaptive crashed semi to **57.52**. Integrity-safe ceiling ≈ **59.2** (match-fair + keep winners); paper **61.36** and probe **60.5** remain out of reach with catalogued lifts. **One Vast command:** `bash scripts/run_phase0_lock_retrain.sh lastshot 16gb`. Recipe map: [`results/adadae_per/thesis/adaptive_recipe_map_57.json`](results/adadae_per/thesis/adaptive_recipe_map_57.json).

Canonical claim under integrity: **beat fair DDAE** on adaptive PER across all 57 ADBench datasets; unsup already passes paper. Paper gap = protocol/table tax (see loop3/phase4 paper-protocol diag ≈ fair).

---

## Hard truth

| Track | Semi PR / ROC | Notes |
|-------|---------------|-------|
| Paper AnoDDAE | **61.36 / 83.17** | Not reachable under integrity with current evidence |
| Probe floor | **60.5** | Likely FAIL after lastshot |
| Match-fair ceiling | **~59.2** | Best evidence ceiling |
| Fair valstop | **58.70 / 82.94** | Integrity twin |
| Disk (pre-lastshot) | **57.52 / 82.44** | Full-57 crash |
| Unsup PER | **33.01 / 75.17** | **PASS** vs paper 32.77 / 74.08 |

---

## Integrity / ship contract

| Rule | Value |
|------|--------|
| Recipe | **One** YAML: `configs/adadae_per.yaml` with `policy: per` |
| Protocol | **570** = 57 × 2 × seeds `{111,222,333,444,555}` |
| Early stop | `val_loss`; never test-PR |
| Adaptive | Per-dataset specialists + upgrades; PHASE0 locks; composed select |
| Ship gate | PR **and** ROC **>** paper on **both** + `G_AP_PR_consistency` |
| Probe floor | Semi PR **≥ 60.5** before `ship` (informational after lastshot) |
| Hardware | **`16gb`** |

---

## Adaptive recipe (all 57)

| Bucket | Datasets | Policy |
|--------|----------|--------|
| PHASE0 lock | wine census FashionMNIST MNIST-C InternetAds optdigits | `baseline_ddae` |
| Composition strip | backdoor thyroid | `baseline_ddae` (no taps/cal_fuse) |
| Protect | smtp donors http musk breastw shuttle Ionosphere Lymphography pendigits magic.gamma vowels skin PageBlocks Stamps WBC Pima letter (+wine/census) | `baseline+protect` |
| Proven lifts | Hepatitis cover glass Waveform cardio satimage-2 speech SVHN satellite | keep RDT/champion/apex/cal_fuse as mapped |
| NLP taps | **Imdb only** | Agnews/Amazon/Yelp/20newsgroups stripped |
| A6 strip | fraud apex/delta, WPBC helix | removed |

Full resolve dump: `python scripts/dump_adaptive_recipe_map.py`

---

## Vast — THE only command (last credit)

```bash
cd /data/ITM/project   # or /workspace/ITM/project
source .venv/bin/activate
export OMP_NUM_THREADS=12
python scripts/detect_hardware.py

# Local/preflight (no GPU):
bash scripts/run_phase0_lock_retrain.sh audit
python scripts/dump_adaptive_recipe_map.py
python scripts/assert_final_config.py --config configs/adadae_per.yaml

# ONE GPU spend — disasters ∪ residuals ∪ strips ∪ proven midtier:
bash scripts/run_phase0_lock_retrain.sh lastshot 16gb
```

Expect semi ~**59.0–59.3**, beat-fair if recovery works; probe 60.5 likely still FAIL. Do **not** run `CONFIRM_FULL57_ALL` or another select.

Optional only if lastshot macros ≥ ~59.0 **and** credit remains:

```bash
# Full semi refresh for clean 570 claim (expensive)
python scripts/invalidate_per_semi_jobs.py --all-semi
bash scripts/run_adadae_per_protocol.sh final 16gb
bash scripts/run_adadae_per_protocol.sh compare
bash scripts/run_adadae_per_protocol.sh gates
```

Ship only if `phase4_probe_gate.json` → `pass: true` (unlikely):

```bash
bash scripts/run_hard_tail_ship_path.sh ship 16gb
```

---

## Do not

- Blind `full57 all` / GPU recipe select
- Re-enable PHASE0 RDT or stripped taps/cal_fuse/apex/helix
- Claim paper-protocol diagnostic as ship
- Spend a second probe retrain after lastshot (probe is read from compare)

---

## Sync back

```bash
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/data/ITM/project/results/adadae_per/ \
  /home/zarnihlawn/Desktop/ITM/project/results/adadae_per/
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/data/ITM/project/configs/adadae_per_*.yaml \
  /home/zarnihlawn/Desktop/ITM/project/configs/
```
