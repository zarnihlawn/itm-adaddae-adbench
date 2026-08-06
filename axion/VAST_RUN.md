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

## Runs (GPU)

```bash
cd /workspace/ITM/project/axion
source .venv/bin/activate
export PYTHONPATH=src PYTHONUNBUFFERED=1

bash scripts/vast_probe.sh smoke
bash scripts/vast_probe.sh classical
bash scripts/vast_probe.sh g2

# Optional:
bash scripts/vast_probe.sh g2-fast
bash scripts/vast_probe.sh g2 configs/gpu.yaml
```

Or raw:

```bash
python scripts/run_probe.py --all-probe --model axion --loop-log --run-id axion_g2
```

## Sync results back (laptop)

```bash
rsync -avz -e "ssh -p PORT" \
  root@sshN.vast.ai:/workspace/ITM/project/axion/results/ \
  /home/zarnihlawn/Desktop/ITM/project/axion/results/
```

## G2 pass rule

Both settings: PR ≥ paper+2 and ROC ≥ paper+1  
Paper: unsup **32.77 / 74.08**, semi **61.36 / 83.17**

Check: `results/axion_g2/probe_summary.json` → `probe_macro.*.pass_probe_margin`

## Local note

- Do not mix with archived DDAE-PAR under `ITM/archive/` or old AdaDDAE `results/adadae_*`
- ADBench path resolves to `ITM/ADBench/...` automatically from `project/axion`
