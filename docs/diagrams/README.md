# AdaDDAE Version Diagrams

Design, architecture, and flow diagrams for every AdaDDAE generation.

| Version | Combined PR | Diagram pack |
|---------|-------------|--------------|
| [DDAE baseline](00_ddae_baseline.md) | 46.69% | Paper reproduction |
| [v2](02_v2.md) | 47.31% | Early hybrid |
| [v2.1](02_1_v2_1.md) | 47.38% | Patched hybrid |
| [v3](03_v3.md) | 48.37% | Routed policies |
| [v3.1](03_1_v31.md) | 49.34% | Unsup fallback + semi tail |
| [v4](04_v4.md) | 49.93% | Semi/unsup tracks |
| [v4.1](04_1_v41.md) | 49.92% | Frozen thesis baseline |
| [v5](05_v5.md) | 49.57% | GATE-only (regressed) |
| [v5.1](05_1_v51.md) | **50.12%** | MCE+SMC+guarded merge |

Also see: [Version evolution overview](00_overview.md)

Each pack typically includes:

1. System / architecture design
2. Training pipeline flow
3. Inference / scoring flow
4. Experiment protocol flow
5. Component stack
6. Delta vs previous version
7. Data / merge flow (where relevant)
