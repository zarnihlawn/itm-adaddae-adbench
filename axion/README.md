# AXION

**A**daptive cross-feature **I**nteraction **O**bservation **N**etwork — tabular AD on ADBench under the **AnoDDAE / Livernoche** protocol.

Location: `ITM/project/axion/` (same git repo as `project`). Does not reuse AdaDDAE-PER / DDAE-PAR code paths.

## Status

- **Phase 0–1:** protocol + harness — done  
- **Phase 2:** AXION model + Vast runners — done (G2 FAIL; see `results/axion_g2/`)  
- **Phase 3:** train-anchored score + v1-balanced hparams — current (re-probe as `axion_g3`)  

**Vast:** [`VAST_RUN.md`](VAST_RUN.md) · **Protocol:** [`PROTOCOL.md`](PROTOCOL.md)

```bash
cd /data/ITM/project/axion   # or /workspace/...
bash scripts/vast_setup.sh
bash scripts/vast_probe.sh g3-classical configs/gpu.yaml   # gate: semi PR ≥ 60
bash scripts/vast_probe.sh g3 configs/gpu.yaml             # full 12-ds probe
# full57 only after both settings pass margin
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
