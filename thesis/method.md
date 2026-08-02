# AdaDDAE Method Notes (Master's Thesis)

## Baseline

**DDAE / DDAE-C** (Sattarov et al., KDD 2025, arXiv:2508.00758) integrate diffusion noise scheduling into a denoising autoencoder for tabular anomaly detection on ADBench (57 datasets), in unsupervised and semi-supervised protocols.

Published means (Table 1, %):

| Setting | Model | PR-AUC | ROC-AUC |
|---------|-------|--------|---------|
| Unsupervised | DDAE | 32.77 ± 0.69 | 74.08 ± 0.51 |
| Semi-supervised | DDAE | 61.36 ± 2.23 | 83.17 ± 1.0 |

## Problem with a fixed global schedule

DDAE uses a schedule that is largely global per learning paradigm. The paper’s own analysis shows:

- Unsupervised: later timesteps and **cosine** scheduling help (noise as regularizer).
- Semi-supervised: moderate \(T \approx 50\text{–}100\) and **linear** scheduling work best.

Their conclusion / future work explicitly calls for **adaptive noise scheduling** and better contrastive pair design. AdaDDAE v2 implements that as a principled framework with SNR-guided DANC and SNR-weighted SCS.

---

## Formal objective (training)

### Forward diffusion

For timestep \(t \in \{1,\ldots,T\}\), with noise schedule \(\beta_t\) and cumulative product \(\bar{\alpha}_t = \prod_{i=1}^{t}(1-\beta_i)\):

\[
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

### Reconstruction loss (DDAE Eq. 5)

AdaDDAE predicts clean data via encoder–decoder bottleneck \(g_\phi(f_\theta(\cdot))\), **not** noise prediction:

\[
\mathcal{L}_{\text{rec}} = \mathbb{E}_{x_0,\epsilon,t}\,\big\|x_0 - g_\phi\big(f_\theta(x_t, e_t)\big)\big\|_2^2
\]

where \(e_t\) is the sinusoidal time embedding.

### Contrastive loss (DDAE-C Eq. 7–8, AdaDDAE v2)

Latent distance \(\delta = \|z_0 - z_t\|_2\) with margin \(m_t = 1 + \gamma \cdot t/T\):

\[
\mathcal{L}_{\text{cont}} = \mathbb{E}\Big[(1-y)\,\delta^2 + y\,\max(0, m_t - \delta)^2\Big]
\]

- **Semi-supervised:** positive pairs \((x_0^{(i)}, x_0^{(j)})\) sampled only from training normals (\(y=0\)).
- **Hard negatives:** same-sample \((x_0, x_t)\) with \(y=1\).

Noise-adaptive mixing (thesis extension over fixed \(\alpha\)):

\[
\alpha_t = \alpha_0 \cdot \frac{t}{T}, \quad
\mathcal{L} = (1-\alpha_t)\,\mathcal{L}_{\text{rec}} + \alpha_t\,\mathcal{L}_{\text{cont}}
\]

---

## DANC v2 — SNR-guided adaptive schedule

**Baseline DDAE:** fixed \((T, \text{scheduler})\) per paradigm.

**AdaDDAE v2:** choose \(T^*\) from a target terminal signal-to-noise ratio \(\tau_{\text{snr}}\):

\[
T^* = \min\big\{T : \bar{\alpha}_T \leq \tau_{\text{snr}}\big\}
\]

| Setting | \(\tau_{\text{snr}}\) (typical) | Scheduler |
|---------|--------------------------------|-----------|
| Unsupervised | \(10^{-4}\)–\(10^{-3}\) (heavy noise) | cosine |
| Semi-supervised | \(0.05\)–\(0.2\) (preserve manifold) | linear |

Meta-features \((n, d, \widehat{\text{contamination}}, \text{skewness}, \text{intrinsic-dim proxy})\) modulate \(\tau_{\text{snr}}\) and \(\beta_{\text{end}}\). GPU profile allows higher \(T_{\max}\) (up to 500) for paper-faithful unsupervised runs.

### LF-DANC (label-free, default)

Contamination \(\hat{c}\) is estimated from train-only robust z-score tail rate (no labels). Config `danc_contamination_mode: label_free | oracle`; oracle uses true \(y\) for ablation upper bound only.

### MANS (Manifold-Aligned Noise Schedule)

\[
\beta_{\text{end}}^* = \beta_0 \cdot \Big(1 + \log(1 + d_{\text{eff}}/d)\Big), \quad
\tau_{\text{snr}}^* = \tau_0 \cdot (1 + \mathbb{1}_{\text{semi}} \cdot \hat{c})
\]

---

## SSTS — SNR-Stratified Timestep Selection

Timesteps \(\mathcal{T}^*\) are chosen by **weighted quantile stratification** on \(\bar{\alpha}_t\) (not uniform linspace). Config: `scs_selection: snr_stratified | linspace`.

---

## TAPS — Timestep-Adaptive Pair Sampling

| Setting | Positives | Negatives |
|---------|-----------|-----------|
| Semi-supervised | \(z_0\) from training normals only | \((z_0, z_t)\) |
| Unsupervised | Same-batch at shared \(t\) | Hard negative at high \(t\) |

\[
\alpha_{\text{eff}} = \alpha_0 \cdot (t/T) \cdot w_t
\]

Config: `contrastive_pairing: taps | random`.

---

## VUS — Variance-Uncertainty Score (4th view)

\[
s_{\text{var}}(x_0) = \sum_{t \in \mathcal{T}^*} w_t \cdot \mathrm{Var}_{\epsilon}\big[\|x_0 - \hat{x}_0(x_t^{(\epsilon)})\|_2\big]
\]

Fused with calibrated \(\lambda_{\text{var}}\). Config: `use_uncertainty_view: true`, `uncertainty_draws: 3`.

---

## RDT — Rejection-aware Diffusion Training

Inspired by [TabADM](https://arxiv.org/abs/2307.12336): down-weight training points with suspiciously high clean reconstruction loss (likely anomalies in unsupervised data).

After warmup epoch, compute per-sample loss at \(t=1\):

\[
w_i = \begin{cases} 1 & \mathcal{L}_{\text{rec}}(x_i) \leq Q_{0.95} \\ \lambda_{\min} & \text{otherwise} \end{cases}
\]

Training loss is weighted: \(\mathcal{L} = \sum_i w_i \ell_i / \sum_i w_i\). Config: `use_rejection_training: true`.

---

## DTE-inspired proxy — time / kNN scoring view (5th view)

Inspired by [DTE (ICLR 2024)](https://arxiv.org/abs/2305.18593): anomalies are off-manifold → higher expected diffusion time. AdaDDAE uses a **proxy**, not the full DTE estimator (see [`src/models/dte.py`](../src/models/dte.py)).

**Soft time estimate from reconstruction:**

\[
\widehat{\mathbb{E}}[t \mid x_0] = \sum_{t \in \mathcal{T}^*} p(t \mid x_0)\, t, \quad p(t \mid x_0) \propto \|x_0 - \hat{x}_0(x_t)\|^2
\]

**kNN latent proxy:** \(s_{\text{DTE}}^{\text{kNN}} \propto d_{\text{kNN}}(z_0)\), fused with the soft time estimate. Config: `use_dte_view: true`. Thesis wording must say “DTE-inspired proxy,” not “we implement DTE.”

Primary recipe: [`configs/adadae_final.yaml`](../configs/adadae_final.yaml). Claims map: [`thesis/claims_code_map.md`](claims_code_map.md).

---

## SCS v2 — SNR-weighted selective scoring

### DDAE (Eq. 6)

\[
S_{\text{DDAE}}(x_0) = \sum_{t=1}^{T} \big\|x_0 - \hat{x}_0(x_t, t)\big\|_2^2
\]

### AdaDDAE v2 (multi-view + SNR weights)

\[
S_{\text{AdaDDAE}}(x_0) = \sum_{t \in \mathcal{T}^*} w_t \Big(
\lambda_r \|x_0 - \hat{x}_0\|_2^2 +
\lambda_z \|z_0 - z_t\|_2^2 +
\lambda_\epsilon \|\epsilon - \hat{\epsilon}\|_2^2
\Big)
\]

Timesteps \(\mathcal{T}^*\) are SNR-stratified samples (up to `scs_max_timesteps`, 64 on GPU). Weights align with paper Fig. 6:

| Setting | \(w_t\) |
|---------|---------|
| Unsupervised | \(\propto (1 - \bar{\alpha}_t)\) — late / noisy steps dominate |
| Semi-supervised | \(\propto \bar{\alpha}_t\) — early / structure-preserving steps dominate |

Ablation flag `scs_mode: full_sum | snr_weighted | uniform`.

### Calibrated fusion (semi-supervised)

Replace fixed \(\lambda_r,\lambda_z,\lambda_\epsilon\) with validation-normal calibration:

\[
\lambda_v = \frac{1}{\mathbb{E}_{x \sim \mathcal{N}}[s_v(x)] + \epsilon}, \quad v \in \{r, z, \epsilon, \text{var}\}
\]

then normalize \(\sum_v \lambda_v = 1\). Unsupervised uses population normalization over the test batch.

---

## FTP — Feature Tuning Pipeline

Leak-safe preprocessing: fit scaler / PCA / clip **only on training normals** in semi-supervised mode.

- Whitening / PCA when \(d \gg n_{\text{eff}}\) reduces manifold dimension.
- PCA cutoff tied to intrinsic-dimension proxy from DANC meta-features.
- Unsupervised: StandardScaler (Robust for heavy skew); semi-supervised: Robust default.

---

## Evaluation protocol (must match paper)

- ADBench 57 datasets (47 Classical + 5 CV ResNet18 families + 5 NLP BERT families)
- Multi-split families (e.g. CIFAR10_0…9): average metrics across splits
- Seeds: `{111, 222, 333, 444, 555}`
- Unsupervised: train and score on all samples
- Semi-supervised: train on 50% of normals; test = remaining normals + all anomalies
- Metrics: PR-AUC, ROC-AUC, AP

**Fair primary comparison:** Table deltas use the same train/test splits, seeds, epochs, LR, val carve (`val_fraction: 0.2`), `early_stop_metric: val_loss`, patience, model width/latent, and base diffusion betas / `time_emb_dim` as [`configs/baselines_ddae_valstop.yaml`](../configs/baselines_ddae_valstop.yaml). Only Ada method flags (DANC, SCS, FTP, fusion, A2–A4 modules, contrastive, etc.) differ. `scripts/assert_final_config.py` locks those shared knobs. Published paper Table-1 means remain a secondary reference via `scripts/compare_to_ddae.py` (not the apples-to-apples baseline).

---

## Hardware profiles

### CPU (laptop)

- Soft RSS limit 8 GiB, `max_train_samples: 50000`, SCS ≤ 32, sequential scoring
- Config: `configs/default.yaml` + `hardware_cpu.yaml`

### GPU (Vast.ai RTX 5070 8GB)

- VRAM soft limit 6.8 GB, AMP bf16, vectorized multi-timestep scoring
- Full training data (`max_train_samples: 0`), SCS up to 64 steps
- Config: `configs/default_gpu.yaml` + `hardware_rtx5070.yaml`
- Bootstrap: `bash scripts/setup_vast.sh`; throughput: `python scripts/benchmark_gpu.py`

Log **rss_mb** and **vram_mb** in JSONL for reproducibility.

---

## Primary thesis claim

**Primary model** = one frozen recipe [`configs/adadae_final.yaml`](../configs/adadae_final.yaml) (`policy: static`, val-only early stop). It improves mean PR-AUC / ROC-AUC over a **fair DDAE baseline** (same val-stop protocol) on ADBench 57×2×5, with ablations isolating LF-DANC, MANS, SSTS, TAPS, VUS, FTP, DTE-proxy, RDT, and calibrated fusion. See [`thesis/novelty.md`](novelty.md) and [`thesis/claims_code_map.md`](claims_code_map.md).

Secondary claim: competitive standing vs diffusion baselines (DDPM, DTE-*). Matching every classical #1 method unsupervised is **not** required for the primary claim. Routed hybrids and guarded merges are **appendix only**.

## Known limitations (disclose)

- Very large tables may be train-subsampled on CPU only; GPU uses full protocol data.
- SCS subset is a deliberate trade-off vs full-sum Eq. 6; ablate `scs_mode: full_sum` on a subset.
- DTE-inspired scoring is a **proxy** (recon-softmax + kNN), not full ICLR 2024 DTE.
- Early ablations on 5 datasets with few epochs showed `full_adadae` below `scs`-only; GPU runs use 100 epochs, calibrated fusion, and SNR-guided schedules to address this.
