# Cross-Version Comparison Diagrams

## 1. Feature matrix

```mermaid
flowchart TB
  subgraph Features
    FTP
    DANC
    SCS
    Route[Routing]
    TAPS
    VUS
    RDT
    DTE
    MCE
    SMC
    GATE
    Guard[RegGuard]
  end
```

| Feature | DDAE | v2 | v2.1 | v3 | v3.1 | v4 | v4.1 | v5 | v5.1 |
|---------|------|----|------|----|------|----|------|----|------|
| FTP | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| LF-DANC / MANS | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SCS / SSTS | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Policy routing | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TAPS / VUS / RDT / DTE | | | | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Patch hybrid merge | | | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ |
| MCE | | | | | | | | planned | ✓ |
| SMC | | | | | | | | planned | ✓ |
| GATE | | | | | | | | ✓ v1 | ✓ v2 |
| Regression guard | | | | | | | | | ✓ |

## 2. Score path evolution

```mermaid
flowchart LR
  DDAE[full_sum_recon] --> V2[SCS_3view]
  V2 --> V3[5view_calibrated]
  V3 --> V41[v4.1_stable_5view]
  V41 --> V5[GATE_rank_blend]
  V41 --> V51[MCE_or_SMC_or_GATE_WTA]
```

## 3. Research risk vs reward

```mermaid
quadrantChart
  title Experiment risk vs macro gain
  x-axis Low risk --> High risk
  y-axis Low gain --> High gain
  quadrant-1 Risky wins
  quadrant-2 Ideal
  quadrant-3 Safe small
  quadrant-4 Avoid
  v2: [0.3, 0.35]
  v3: [0.45, 0.55]
  v31: [0.4, 0.7]
  v4: [0.5, 0.8]
  v41: [0.25, 0.75]
  v5: [0.85, 0.2]
  v51: [0.55, 0.85]
```

## 4. End-to-end product view (v5.1)

```mermaid
C4Context
  title AdaDDAE Context
  Person(researcher, "Researcher", "Runs protocols, writes thesis")
  System(adadae, "AdaDDAE", "Adaptive diffusion AD framework")
  System_Ext(adbench, "ADBench", "57 anomaly datasets")
  System_Ext(vast, "Vast GPU", "Training jobs")
  Rel(researcher, adadae, "configure & validate")
  Rel(adadae, adbench, "loads npz splits")
  Rel(adadae, vast, "executes GPU tracks")
```

## 5. Job accounting

```mermaid
flowchart TB
  Full[Full_hybrid_570]
  Patch[Patch_tracks_variable]
  Full --> Report[Table1_macro]
  Patch --> Guard[guarded_promotion]
  Guard --> Full
```
