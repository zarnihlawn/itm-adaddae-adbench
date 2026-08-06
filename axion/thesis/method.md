# AXION method (Phase 2–3)

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
| **SCALE** | Train-only `n,d` → hidden / depth / K / mask rates / high-mask bank |

## Training

- Loss: Gaussian NLL on masked cells only
- Early stop: val NLL from train carve (`val_loss`) — never test PR
- GPU: AMP fp16 + larger batch

## Score (Phase 3)

Train-anchored z-norm (fit MCS/LATCH mean+std on train; apply at score time — never test-batch z):

\[
s(x)=\mathrm{z}_{\mathrm{train}}(\mathrm{MCS}(x))+\alpha\,\mathrm{z}_{\mathrm{train}}(\mathrm{LATCH}(x))
\]

- Default \(\alpha=0.40\) (unsupervised / mixed train)
- Semi (train all-normal): \(\alpha=\texttt{latch\_alpha\_semi}=0.25\)
- Defaults: `mae_weight=0.60`, `nll_weight=0.40`
- High-\(d\) (\(d\ge400\)): higher `score_k`, softer high-mask bank (`+0.15`, cap `0.6`), dropout `0.05`

## Protocol

AnoDDAE paper-faithful splits (`PROTOCOL.md`). Not ADBench `DataGenerator`.
