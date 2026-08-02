# DDAE Baseline — Diagram Pack

**Role:** Paper reproduction / G4 reference  
**Combined PR:** 46.69% (backup) · Paper Table-1: 47.07%  
**Artifact:** `backup/ddae_baseline_570/`

---

## 1. System architecture

```mermaid
flowchart TB
  subgraph Input
    X[ADBench_X]
    Y[labels_eval_only]
  end
  subgraph Pre
    Scale[StandardScaler]
  end
  subgraph Model
    Enc[Encoder]
    Lat[Latent_z]
    Dec[Decoder]
    Diff[Fixed_diffusion_schedule]
  end
  subgraph Score
    Rec[Reconstruction_sum_over_t]
  end
  X --> Scale --> Enc --> Lat --> Dec
  Diff --> Enc
  Diff --> Dec
  Dec --> Rec
  Y -.->|test_metrics| Rec
```

## 2. Training flow

```mermaid
sequenceDiagram
  participant D as Dataset
  participant S as Scaler
  participant M as DDAE
  participant L as Loss

  D->>S: fit_transform(train)
  loop epochs
    S->>M: batch_x
    M->>M: sample_t ~ Uniform
    M->>M: add_noise(x,t)
    M->>L: recon + optional_contrastive
    L->>M: backprop
  end
```

## 3. Inference / scoring flow

```mermaid
flowchart LR
  Xtest[X_test] --> Scale
  Scale --> Loop{for_all_t}
  Loop --> Noise[corrupt_x_t]
  Noise --> Pred[predict_x0]
  Pred --> Err[||x - xhat||]
  Err --> Sum[sum_over_t]
  Sum --> Rank[anomaly_rank]
```

## 4. Experiment protocol

```mermaid
flowchart TB
  A[57_datasets] --> B[2_settings]
  B --> C[5_seeds]
  C --> D[570_jobs]
  D --> E[macro_PR_ROC]
  E --> F[compare_to_paper]
```

## 5. Component stack

```mermaid
block-beta
  columns 1
  block:stack
    columns 1
    A["Fixed β schedule"]
    B["Full-sum reconstruction score"]
    C["Optional contrastive (DDAE-C)"]
    D["No FTP / DANC / routing"]
  end
```

## 6. Design limitation (motivates AdaDDAE)

```mermaid
flowchart LR
  Fixed[One_global_schedule]
  Hetero[57_heterogeneous_datasets]
  Gap[Suboptimal_noise_per_dataset]
  Future[Paper_calls_for_adaptive_scheduling]

  Fixed --> Gap
  Hetero --> Gap
  Gap --> Future
```

## 7. Data path on Vast

```mermaid
flowchart LR
  ADBench[ADBench/datasets/*.npz]
  Run[run_one.py]
  Out[backup/ddae_baseline_570]
  ADBench --> Run --> Out
```
