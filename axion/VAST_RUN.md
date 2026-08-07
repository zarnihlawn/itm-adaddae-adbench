# AXION on Vast.ai

Lives inside the git repo at `ITM/project/axion/` (push/pull with `project`).

Layout on the instance:

```
/workspace/ITM/          # or /data/ITM/
  ADBench/adbench/datasets/
  project/               # git clone / pull
    axion/               # this package
```

## One-time setup

```bash
cd /workspace/ITM/project
git pull
cd axion
bash scripts/vast_setup.sh
```

## Runs (GPU) — Phase 4

```bash
cd /workspace/ITM/project/axion
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

# Preferred: classical → hard gate → full (stops if semi PR < 60)
bash scripts/vast_probe.sh g4-auto configs/gpu.yaml

# Or step-by-step:
bash scripts/vast_probe.sh g4-classical configs/gpu.yaml
# read MACRO; only if semi ≥ 60:
bash scripts/vast_probe.sh g4 configs/gpu.yaml   # refuses to start without gate
```

Do **not** use bare `g3-classical && g3` — that only skips on crash, not on bad MACRO.

## Sync results back (laptop)

```bash
rsync -avz -e "ssh -p PORT" \
  root@HOST:/data/ITM/project/axion/results/ \
  /home/zarnihlawn/Desktop/ITM/project/axion/results/
```

## Pass rule

Both settings: PR ≥ paper+2 and ROC ≥ paper+1  
Paper: unsup **32.77 / 74.08**, semi **61.36 / 83.17**

Classical glance gate (before full probe): semi PR **≥ 60**  
Check: `results/axion_g4/probe_summary.json` → `probe_macro.*.pass_probe_margin`  

Do **not** run `full57` until both settings pass (script also checks).

## Local note

- Do not mix with archived DDAE-PAR under `ITM/archive/` or old AdaDDAE `results/adadae_*`
- ADBench path resolves to `ITM/ADBench/...` automatically from `project/axion`
