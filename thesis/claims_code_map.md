# Claims ↔ code map (primary recipe = `configs/adadae_final.yaml`)

| Claim | What it is | Code | Config flag | Ablation note |
|-------|------------|------|-------------|---------------|
| FTP | Leak-safe scaler/PCA fit on train only | `src/features/pipeline.py` | `adadae.use_ftp` | vs `use_ftp: false` |
| LF-DANC | Label-free schedule from train meta | `src/models/danc.py` | `adadae.use_danc` + `danc_contamination_mode: label_free` | vs fixed `diffusion.*`; oracle is ablation-only |
| MANS | β_end from effective dimension | `src/models/danc.py` (`danc_policy`) | part of DANC | coupled with LF-DANC |
| SSTS / SCS | SNR-stratified timesteps + weights | `src/models/scs.py` | `adadae.use_scs`, `scs_selection` | vs full-sum / linspace |
| TAPS | Timestep-adaptive contrastive pairs | `src/models/adadae.py` | `train.contrastive`, `contrastive_pairing: taps` | off in DDAE baseline |
| VUS | Variance-over-noise scoring view | `src/models/adadae.py` `_score_views` | `use_uncertainty_view` | optional view; often off in AdaDDAE-2 |
| DTE-inspired proxy | Soft E[t\|x] from recon + kNN latent distance — **not** full ICLR 2024 DTE | `src/models/dte.py` | `use_dte_view` | call it proxy in thesis |
| RDT | Train loss quantile down-weighting | `src/models/adadae.py` | `use_rejection_training` | TabADM-inspired |
| Calibrated fusion | Train-normal view scale calibration | `src/models/adadae.py` `_calibrate_fusion` | `fusion_mode: calibrated` | vs fixed / SMC (appendix) |
| Val-only early stop | Checkpoint on carved val recon loss; test once | `carve_val_from_train`, `AdaDDAE.fit` | `train.val_fraction`, `early_stop_metric: val_loss` | never `y_test` in fit |
| One frozen recipe | Same YAML for all 57 datasets | `configs/adadae_final.yaml` | `adadae.policy: static` | routing / exceptions = appendix |

## AdaDDAE-2 (Table 2)

| Claim | What it is | Code | Config flag |
|-------|------------|------|-------------|
| CHRONOS | Shared schedule hypernet from train meta | `src/models/adadae2/chronos.py` | `use_chronos` |
| GEODE | Local PCA off-manifold residual in z | `src/models/adadae2/geode.py` | `use_geode` |
| CALIX | Conformal/quantile view fusion | `src/models/adadae2/calix.py` | `fusion_mode: calix` |
| NEXUS | VICReg SSL on tabular augmentations | `src/models/adadae2/nexus.py` | `use_nexus` |
| AETHER | DSM train loss + energy path score | `src/models/adadae2/aether.py` | `use_aether` |
| Frozen AdaDDAE-2 | One recipe all 57 | `configs/adadae2_final.yaml` | `policy: static`, VUS off |

## AdaDDAE-3 (Table 3)

| Claim | What it is | Code | Config flag |
|-------|------------|------|-------------|
| HELIOS | Continuous SNR time map (monotonic MLP) | `src/models/adadae3/helios.py` | `use_helios` |
| KAIROS | Dual train-T curriculum vs score budget | `src/models/adadae3/kairos.py` | `use_kairos` |
| ORBIS | FFT top-k residual spectral energy | `src/models/adadae3/orbis.py` | `use_orbis` |
| STRATA | Multi-scale latent pyramid consistency | `src/models/adadae3/strata.py` | `use_strata` |
| PLEXUS | Graph-kNN message residual on memory | `src/models/adadae3/plexus.py` | `use_plexus` |
| PHASOR | Amplitude-gated sin/cos time emb | `src/models/adadae3/phasor.py` | `use_phasor` / `time_emb_type: phasor` |
| ARGOS | Evidential / conflict-aware fusion | `src/models/adadae3/argos.py` | `fusion_mode: argos` |
| AEGIS | Split-conformal + temperature scale | `src/models/adadae3/aegis.py` | `use_aegis` |
| MIRAGE | CPU-safe epistemic variance (vs VUS) | `src/models/adadae3/mirage.py` | `use_mirage` |
| NEXUS-v2 | Barlow + VICReg SSL | `src/models/adadae3/nexus_v2.py` | `use_nexus_v2` |
| RDT-v2 | Soft logistic rejection + EMA | `src/models/adadae3/rdt_v2.py` | `use_rdt_v2` |
| LYNX | Rank aggregation across views | `src/models/adadae3/lynx.py` | `fusion_mode: lynx` |
| ATLAS | FiLM / AdaLN from meta-φ | `src/models/adadae3/atlas.py` + `network.py` | `use_atlas` |
| HYDRA | Multi-head trunk | `src/models/adadae3/hydra.py` | `use_hydra` |
| FLUX | Light flow-matching head | `src/models/adadae3/flux.py` | `use_flux` |
| SCRIBE | SAM-lite every k epochs | `src/models/adadae3/scribe.py` | `use_scribe` |
| EPOCHÉ | Val plateau → LR shrink / SCS expand | `src/models/adadae3/epoche.py` | `use_epoche` |
| Frozen AdaDDAE-3 | One recipe all 57 | `configs/adadae3_final.yaml` | `policy: static`, GATE/VUS off |

### AdaDDAE-3 negative results (smoke triage)

| Flag | Note |
|------|------|
| MIRAGE + FLUX + ATLAS + AEGIS + ARGOS kitchen-sink | Cardio smoke semi PR collapsed (~0.12 vs Table1 ~0.72); demoted in patched `adadae3_final.yaml` |
| Prefer | HELIOS+KAIROS+ORBIS+PLEXUS+NEXUS-v2+RDT-v2+EPOCHÉ + calix |

## AdaDDAE-4 (Table 4)

| Claim | What it is | Code | Config flag |
|-------|------------|------|-------------|
| OMNI | Rich train-only meta + regime tags | `src/models/adadae4/omni.py` | `use_omni`, `auto_regime_gates` |
| NANO | Tiny-n capacity / neighbor shrink | `adadae4/nano.py` | `use_nano` |
| TORRENT | Huge-n memory / batch caps | `adadae4/torrent.py` | `use_torrent` |
| PRISM | Whitened subspace residual (high-d) | `adadae4/prism.py` | `use_prism` |
| POLIS | Multi-prototype latent memory | `adadae4/polis.py` | `use_polis` |
| SIEVE | Contam-aware RDT + IF train view | `adadae4/sieve.py` | `use_sieve` |
| NEEDLE | Rare-c tail boost / conformal α | `adadae4/needle.py` | `use_needle` |
| SPARSE | Jaccard presence view | `adadae4/sparse.py` | `use_sparse_view` |
| ROBUST | MAD-normalize views | `adadae4/robust.py` | `use_robust` |
| QUELL | Conflict-aware view reliability | `adadae4/quell.py` | `fusion_mode: quell` |
| Frozen AdaDDAE-4 | One recipe + train-meta gates | `configs/adadae4_final.yaml` | `policy: static` |
| Regime audit | ADBench CSV tags | `scripts/adbench_regime_audit.py` | — |

## AdaDDAE-5 (Table 5)

Information-geometric stack on **AdaDDAE Table-1 core** (not A3/A4 kitchen-sink). Protocol knobs locked to fair DDAE.

| Claim | What it is | Code | Config flag |
|-------|------------|------|-------------|
| FIGARO | Data-dependent path-energy schedule refine; T★ is a **lower bound** after SNR resolve (DANC no longer fail-opens to T=5) | `adadae5/figaro.py` + `danc._resolve_T_from_snr` | `use_figaro` |
| DSM+ | Joint recon + ε score matching | `adadae5/dsm_plus.py` | `use_dsm_plus` |
| MAHALA | Ledoit–Wolf Mahalanobis on [r;z] (train mean/std locked) | `adadae5/mahala.py` | `use_mahala` |
| FULL-DTE | Sharpened soft p(t\|x) + kNN — **proxy**, same polarity as baseline DTE | `adadae5/full_dte.py` | `use_full_dte` |
| LEXICON | Spearman-agreement fusion priors (Dirichlet-like normalize) | `adadae5/lexicon.py` | `fusion_mode: lexicon` |
| PURA | nnPU-style contamination weights | `adadae5/pura.py` | `use_pura` |
| EVT-TAIL | GPD peaks-over-threshold on **fused** train scores | `adadae5/evt_tail.py` | `use_evt_tail` |
| CONFAL | Empirical CDF on **EVT-transformed** train scores when both on (compose matches predict) | `adadae5/confal.py` | `use_confal` |
| SPECTRA | Laplacian eigenmap blend projected back to original `d` (no capacity leak) | `adadae5/spectra.py` | `use_spectra` |
| SINKHORN | 1D Wasserstein sorted-mass geometry (`d≤64`) | `adadae5/sinkhorn.py` | `use_sinkhorn` |
| IB-LATENT | Compression `β‖z‖²` only | `adadae5/ib_latent.py` | `use_ib_latent` |
| ELBO-S | ε-residual ELBO proxy (ablation-only; **demoted** — was residual clone in Lexicon) | `adadae5/elbo_s.py` | `use_elbo_s: false` |
| CURRICULUM-SNR | Epoch-wise t_max curriculum | `adadae5/curriculum.py` | `use_curriculum_snr` |
| vMF-Z | Hyperspherical latent concentration (+ memory bank) | `adadae5/vmf_z.py` | `use_vmf_z` |
| GEODE/ORBIS | Selective A3 keepers via `auto_regime_gates` (high-d / classical-hard only) | `adadae4/omni.regime_gate_flags` | `auto_regime_gates: true` |
| Frozen AdaDDAE-5 | One recipe all 57 | `configs/adadae5_final.yaml` | `policy: static` |

### AdaDDAE-5 smoke triage (Wave-6)

| Setting | Note |
|---------|------|
| DANC T | Semi + linear fail-closed to `T_max` (not floor 5); FIGARO T★ meaningful |
| EVT→CONFAL | Fit CONFAL on EVT(fused); predict applies EVT then CONFAL |
| ELBO | Default off; Lexicon rebalanced (no `elbo` mass) |
| Easy smoke | cardio/glass/vertebral × 2 settings × 2 seeds |
| Hard smoke | `smoke_hard`: thyroid/letter/speech, PR floor 0.01 (speech ~1.6% contam) |
| Protocol | A5 ablations + LOO wired; subset/loo fail loud (no `\|\| true`) |

Metric ladder (full 570): G5-1 vs fair DDAE → G5-2/G5-3 absolute; unsup PR ≥80 macro = moonshot (disclose).

**Re-run note:** DANC T fix changes realized timesteps for A1–A5 semi → re-smoke Table 1 after Wave-6; Vast order still Table 1 → 2 → 3 → 4 → 5 → **6**.

## AdaDDAE-6 (Table 6)

ADBench 10-loop regime stack on **A5 integrity core** (see [`adbench_improve_catalog.md`](adbench_improve_catalog.md)). Not A3/A4 kitchen-sink.

| Claim | What it is | Code | Config flag |
|-------|------------|------|-------------|
| HELIX | Train-only linear vs cosine schedule family | `adadae6/helix.py` | `use_helix` |
| DELTA | LF contam sandwich → τ retune | `adadae6/delta.py` | `use_delta` |
| APEX | Contam-aware rare-tail / log-odds map (after EVT→CONFAL) | `adadae6/apex.py` | `use_apex` |
| NAUTILUS | Tiny-n capacity / kNN / memory shrink | `adadae6/nautilus.py` | `use_nautilus` |
| TORQUE | Huge-n bank/batch caps | `adadae6/torque.py` | `use_torque` |
| ORBIT | Cosine / whitened residual view | `adadae6/orbit.py` | `use_orbit` |
| KALE | Conflict-aware fusion weights | `adadae6/kale.py` | `use_kale` / `fusion_mode: kale` |
| RIDGE | Huber recon (+ DSM+ blend) | `adadae6/ridge.py` | `use_ridge` |
| LOCUS | Latent LOF density-ratio view | `adadae6/locus.py` | `use_locus` |
| SPIRAL | One-step reverse consistency score | `adadae6/spiral.py` | `use_spiral` |
| Frozen AdaDDAE-6 | One recipe all 57 | `configs/adadae6_final.yaml` | `policy: static` |

Honesty: APEX is not formal conformal coverage; LOCUS is a LOF proxy; SPIRAL is one-step reverse, not full ODE.

### AdaDDAE-6 smoke triage

| Setting | Note |
|---------|------|
| Easy smoke | **12/12 PASS** cardio/glass/vertebral; cardio semi PR ≈ 0.68–0.71 |
| Hard smoke | **12/12 PASS** thyroid/letter/speech (PR floor 0.01) |
| Fusion | Default `kale` (conflict-aware); Lexicon available via LOO |

## Not primary (appendix / development history only)

| Artifact | Why demoted |
|----------|-------------|
| `policy: routed` + `policy_exceptions.yaml` | Per-dataset specialists |
| `merge_v5_guarded.py` / hybrid completed.json | Test-PR gated cherry-pick |
| `use_mce` / `use_gate` / `fusion_mode: smc` | Optional v5 tracks, not the frozen model |
| Legacy `results/adadae_v*_hybrid` | Development backups — never primary Table 1 |
