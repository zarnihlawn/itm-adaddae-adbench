# AXION

**A**daptive cross-feature **I**nteraction **O**bservation **N**etwork — tabular AD on ADBench under the **AnoDDAE / Livernoche** protocol.

Location: `ITM/project/axion/` (same git repo as `project`). Does not reuse AdaDDAE-PER / DDAE-PAR code paths.

## Status

- **Phase 0–1:** protocol + harness — done  
- **Phase 2:** AXION model + Vast runners — done (G2 FAIL)  
- **Phase 3:** train-anchored + v1 hparams — done (G3 FAIL; classical semi 57.85)  
- **Phase 4:** MCS-primary semi + stronger high-d SCALE + **hard classical gate** — current  

**Vast:** [`VAST_RUN.md`](VAST_RUN.md) · **Protocol:** [`PROTOCOL.md`](PROTOCOL.md)

```bash
cd /data/ITM/project/axion   # or /workspace/...
git pull && bash scripts/vast_setup.sh   # if needed
# ONE command — classical first; full probe only if semi PR ≥ 60
bash scripts/vast_probe.sh g4-auto configs/gpu.yaml
# full57 only after g4 both margins pass (script enforces)
```

## Layout

```
project/
  axion/
    VAST_RUN.md
    PROTOCOL.md
    configs/{default,gpu}.yaml
    src/axion/{data,eval,models,train}/
    scripts/
    results/
```
