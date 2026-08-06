# AXION method (Phase 2)

**AXION** = Adaptive cross-feature Interaction Observation Network.

Orthogonal to AnoDDAE (no diffusion schedule, no full-sum recon-over-t, no DDAE-C contrastive).

## Components

| Acronym | Role |
|---------|------|
| **MCB** | Mask Curriculum Bank — Bernoulli + block masks; multi-rate |
| **FX-Enc** | Residual MLP on `[x ⊙ (1−m) ‖ m]` |
| **HPD** | Heteroscedastic head `(μ, log σ²)` for masked cells |
| **MCS** | Monte-Carlo score: hybrid MAE+NLL over K masks × dual rate banks |
| **LATCH** | Diagonal Mahalanobis on fully-visible latent `z` |
| **SCALE** | Train-only `n,d` → hidden / depth / K / mask rates |

## Training

- Loss: Gaussian NLL on masked cells only
- Early stop: val NLL from train carve (`val_loss`) — never test PR
- GPU: AMP fp16 + larger batch

## Score

\[
s(x)=\mathrm{z}(\mathrm{MCS}(x))+\alpha\,\mathrm{z}(\mathrm{LATCH}(x))
\]

## Protocol

AnoDDAE paper-faithful splits (`PROTOCOL.md`). Not ADBench `DataGenerator`.
