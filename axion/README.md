# AXION

**A**daptive cross-feature **I**nteraction **O**bservation **N**etwork — tabular AD on ADBench under the **AnoDDAE / Livernoche** protocol.

Location: `ITM/project/axion/` (same git repo as `project`). Does not reuse AdaDDAE-PER / DDAE-PAR code paths.

## Status

- **Phase 0–1:** protocol + harness — done  
- **Phase 2:** AXION model + Vast runners — current  

**Vast:** [`VAST_RUN.md`](VAST_RUN.md) · **Protocol:** [`PROTOCOL.md`](PROTOCOL.md)

```bash
cd /workspace/ITM/project/axion   # after git pull
bash scripts/vast_setup.sh
bash scripts/vast_probe.sh smoke
bash scripts/vast_probe.sh classical
bash scripts/vast_probe.sh g2
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
