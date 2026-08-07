# AXION method (Phase 2–4)

**AXION** = Adaptive cross-feature Interaction Observation Network.

Orthogonal to AnoDDAE (no diffusion schedule, no full-sum recon-over-t, no DDAE-C contrastive).

## Components

| Acronym | Role |
|---------|------|
| **MCB** | Mask Curriculum Bank — Bernoulli + block masks; multi-rate |
| **FX-Enc** | Residual MLP on `[x ⊙ (1−m) ‖ m]` |
| **HPD** | Heteroscedastic head `(μ, log σ²)` for masked cells |
| **MCS** | Monte-Carlo score: hybrid MAE+NLL over K masks × rate banks |
| **LATCH** | Diagonal Mahalanobis on fully-visible latent `z` |
| **SCALE** | Train-only `n,d` → hidden / depth / K / mask rates / high-mask bank |

## Training

- Loss: Gaussian NLL on masked cells only
- Early stop: val NLL from train carve (`val_loss`) — never test PR
- GPU: AMP fp16 + larger batch
- **Semi (all-normal train):** softer mask rates, `semi_epoch_boost` on epochs/patience

## Score (Phase 4)

Train-anchored z-norm (fit MCS/LATCH mean+std on train; apply at score time — never test-batch z):

\[
s(x)=\mathrm{z}_{\mathrm{train}}(\mathrm{MCS}(x))+\alpha\,\mathrm{z}_{\mathrm{train}}(\mathrm{LATCH}(x))
\]

| Setting | \(\alpha\) | Hybrid |
|---------|------------|--------|
| Unsupervised / mixed train | `latch_alpha=0.40` | mae 0.60 / nll 0.40 |
| Semi (all-normal) | **`latch_alpha_semi=0.0` (MCS-only)** | mae 0.80 / nll 0.20 |

- Extra light-mask score bank (anomalies fail light recon under normal-only models)
- High-\(d\) (\(d\ge400\)): hidden 512 / latent 128 / `score_k=24`, soft high-mask cap `0.45`

## Why Phase 4 (vs G3)

G3 classical semi PR **57.85** (miss ≥60); full G3 semi **47.57**. G3 used `latch_alpha_semi=0.25` and was **worse** than v1 classical (59.80). Phase 4 drops semi LATCH, strengthens MCS + high-d SCALE, and **hard-gates** full probes.

## Protocol

AnoDDAE paper-faithful splits (`PROTOCOL.md`). Not ADBench `DataGenerator`.
