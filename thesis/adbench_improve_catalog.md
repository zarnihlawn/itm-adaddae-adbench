# ADBench × AdaDDAE improvement catalog (10 analysis loops)

Train-only adaptive improvements for AdaDDAE after Wave-6. Maps ADBench regimes
to AdaDDAE-6 modules. Integrity lock unchanged: `policy: static`, `val_loss`
early stop, protocol knobs locked to fair DDAE.

## Regime summary (57 datasets)

| Tag | Rule (approx.) | Examples |
|-----|----------------|----------|
| tiny | n &lt; 300 | glass, wine, Hepatitis, MVTec-AD |
| huge | n ≥ 50k | fraud, http, census, donors |
| highd | d ≥ 128 or CV/NLP | speech, InternetAds, ResNet/BERT embeds |
| heavy | contam ≥ 0.2 | SpamBase, satellite, yeast |
| rare | contam ≤ 0.02 | speech, fraud, smtp, cover |
| sparse | nnz fraction &lt; 0.3 | backdoor, census, InternetAds |
| multi | cluster_sep &gt; 1.5 | skin, smtp, Wilt |
| skew | skew ≥ 5 | ALOI, PageBlocks, shuttle |

## Ten loops → A6 modules

| Loop | ADBench focus | Gap vs A1–A5 | Module | Role |
|------|---------------|--------------|--------|------|
| 1 | Unsup vs semi schedule family | FIGARO only refines T*, β* | **HELIX** | Train-only linear vs cosine via path energy |
| 2 | Rare contam (speech, fraud, …) | EVT→CONFAL weak on PR tails | **APEX** | Contam-aware rare-tail / log-odds map |
| 3 | LF-DANC vs true contam mismatch | ĉ bias warps τ_snr | **DELTA** | Sandwich/clip ĉ̃ → retune τ |
| 4 | Tiny-n | Mahala/vMF unstable | **NAUTILUS** | Shrink h / kNN / memory ∝ √n |
| 5 | Huge-n | Bank/FTP cost | **TORQUE** | Subsample + bank/batch caps |
| 6 | CV/NLP embeds | Tabular L2 only | **ORBIT** | Cosine / whitened residual view |
| 7 | Multi-view conflict | Lexicon = Spearman only | **KALE** | Conflict-aware view weights |
| 8 | Heavy / skew | L2 dominated by outliers | **RIDGE** | Huber recon (+ keep DSM+) |
| 9 | Classical mid/hard | No density-ratio view | **LOCUS** | LOF on train latent memory |
| 10 | Path consistency | Forward recon only | **SPIRAL** | One-step reverse consistency |

## Explicit skips (rejected in loops)

- Full A4 always-on / A3 kitchen-sink (MIRAGE+FLUX+ATLAS+AEGIS)
- Formal conformal coverage claims; full ICLR DTE claims
- ELBO-S as Lexicon fusion mass
- Routed policy, GATE/MCE/SMC, oracle contamination, test-PR early stop

## Frozen recipe

Primary: `configs/adadae6_final.yaml` on **A5 integrity core + A6 modules**.
Protocol: `scripts/run_adadae6_protocol.sh`. Vast order: Table 1→2→3→4→5→**6**.
