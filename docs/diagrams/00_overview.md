# AdaDDAE — Version Evolution Overview

## 1. Combined PR climb

```mermaid
xychart-beta
  title "Combined macro PR-AUC (%)"
  x-axis ["DDAE", "v2", "v2.1", "v3", "v3.1", "v4", "v4.1", "v5", "v5.1"]
  y-axis "Combined PR %" 46 --> 51
  bar [46.69, 47.31, 47.38, 48.37, 49.34, 49.93, 49.92, 49.57, 50.12]
```

## 2. Version lineage

```mermaid
flowchart LR
  DDAE[DDAE_baseline_46.69]
  V2[v2_47.31]
  V21[v2.1_47.38]
  V3[v3_48.37]
  V31[v3.1_49.34]
  V4[v4_49.93]
  V41[v4.1_49.92]
  V5[v5_49.57_FAIL]
  V51[v5.1_50.12_PRIMARY]

  DDAE --> V2 --> V21 --> V3 --> V31 --> V4 --> V41
  V41 --> V5
  V41 --> V51
  V5 -.->|negative_result| V51
```

## 3. Capability unlock timeline

```mermaid
timeline
  title AdaDDAE capability unlocks
  section Foundation
    DDAE : Fixed diffusion schedule
         : Reconstruction scoring
  section Named components
    v2 : FTP + DANC + SCS
       : Multiview fusion start
    v2.1 : Patch merge hygiene
  section Routing
    v3 : Policy routing
       : TAPS / VUS / RDT / DTE
    v3.1 : Unsup fallback + semi tail patches
  section Ceiling push
    v4 : Dedicated semi/unsup tracks
    v4.1 : Frozen thesis baseline
  section Adaptive tracks
    v5 : GATE ensemble attempt
    v5.1 : MCE + SMC + guarded merge
```

## 4. What each version owns

```mermaid
mindmap
  root((AdaDDAE))
    DDAE
      Fixed schedule
      Full-sum scoring
    v2_v21
      FTP
      LF-DANC
      SCS
    v3_v31
      Routed policies
      TAPS
      VUS
      RDT
      DTE-View
      Patch hybrid
    v4_v41
      Tracked unsup/semi
      Frozen baseline
    v5
      GATE v1
      Unguarded merge
    v51
      MCE
      SMC
      GATE v2 selective
      Regression guard
```

## 5. Research workflow (all versions)

```mermaid
flowchart TB
  subgraph Local [Laptop]
    Code[code_configs]
    Thesis[thesis_artifacts]
  end
  subgraph Vast [GPU_Vast]
    Smoke[vast_smoke]
    Proto[version_protocol]
    Jobs[570_or_patch_jobs]
  end
  subgraph Merge [CPU_merge]
    Hybrid[hybrid_completed.json]
    Gates[validate_gates]
  end
  Code -->|git_push| Vast
  Smoke --> Proto --> Jobs
  Jobs -->|git_push_results| Local
  Local --> Hybrid --> Gates --> Thesis
```

## 6. Evaluation contract (shared)

```mermaid
flowchart LR
  Jobs[completed.json_570]
  Macro[Table1_macro_mean]
  Unsup[Unsup_PR_ROC]
  Semi[Semi_PR_ROC]
  Comb[Combined_PR]
  G[Gates_G1_G9]

  Jobs --> Macro
  Macro --> Unsup
  Macro --> Semi
  Unsup --> Comb
  Semi --> Comb
  Comb --> G
```
