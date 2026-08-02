# PG Dissertation/Project
# Supervision Meeting Record

**Student:** Zarni Hlawn  
**Student Number:** 2591325  
**Supervisor:** Dr. La Wynn Sandi  
**Project Title:** AdaDDAE: Adaptive Diffusion for Anomaly Detection

| Contact Type | Date | Remarks (main issues arising and any action points) | Initials |
|---|---|---|---|
| 1:1 | 24 June 2026 | **Issue:** The initial proposal was framed in an application-oriented manner rather than as a formal research project. Ambiguous scope and limited research framing reduced academic clarity. **Action:** Revise the project direction toward a clearly defined research topic. Search the resources recommended by the supervisor, review recent scholarly literature, and propose a refined research focus for approval. | |
| 1:1 | 29 June 2026 | **Issue:** A change of topic was required to align the work with a research contribution rather than application development. **Action:** Adopt AdaDDAE (Adaptive Diffusion for Anomaly Detection) as the project focus. Present the research idea, baseline paper, related references, dataset usage, and publicly available code. Reproduce the DDAE baseline and implement the first adaptive components: feature tuning (FTP), label-free dataset-adaptive noise control (LF-DANC / MANS), and SNR-based scoring with early multiview fusion (SCS / SSTS), including pipeline stability refinements. | |
| 1:1 | 13 July 2026 | **Issue:** Early experimental results were approved, but stronger baselines, bug fixes, and a clearer distinction between the proposed novelty and the baseline method were still required. **Action:** Strengthen comparative baselines, resolve identified implementation issues, and clarify the novelty contribution relative to DDAE. Finalise and present the next method stages for approval: policy routing with the named component stack (TAPS, VUS, RDT, DTE-View), unsupervised fallback and semi-supervised tail refinement, and dedicated unsupervised and semi-supervised training tracks. | |
| 1:1 | 23 July 2026 | **Issue:** Completed method stages were reviewed; further extension was needed to improve performance beyond the frozen reference configuration. **Action:** Extend the framework with additional adaptive scoring and merge strategies, including multi-condition and selective ensemble approaches, using regression-guarded promotion so that only non-regressing updates are retained. Prepare comparative metrics and findings for the next review. | |
| 1:1 | 28 July 2026 | **Issue:** The latest adaptive scoring and guarded-merge experiments were approved; formal documentation had not yet been prioritised. **Action:** Begin formal project documentation. Organise methodology, results, figures, and tables into a coherent draft, with emphasis on clarity and consistency in presenting methods and findings. | |

---

## Meeting detail notes

### 24 June 2026
Issue: The initial proposal was framed in an application-oriented manner rather than as a formal research project. Ambiguous scope and limited research framing reduced academic clarity.  
Action: Revise the project direction toward a clearly defined research topic. Search the resources recommended by the supervisor, review recent scholarly literature, and propose a refined research focus for approval.

### 29 June 2026
Issue: A change of topic was required to align the work with a research contribution rather than application development.  
Action: Adopt AdaDDAE (Adaptive Diffusion for Anomaly Detection) as the project focus. Present the research idea, baseline paper, related references, dataset usage, and publicly available code. Reproduce the DDAE baseline and implement the first adaptive components: feature tuning (FTP), label-free dataset-adaptive noise control (LF-DANC / MANS), and SNR-based scoring with early multiview fusion (SCS / SSTS), including pipeline stability refinements.

### 13 July 2026
Issue: Early experimental results were approved, but stronger baselines, bug fixes, and a clearer distinction between the proposed novelty and the baseline method were still required.  
Action: Strengthen comparative baselines, resolve identified implementation issues, and clarify the novelty contribution relative to DDAE. Finalise and present the next method stages for approval: policy routing with the named component stack (TAPS, VUS, RDT, DTE-View), unsupervised fallback and semi-supervised tail refinement, and dedicated unsupervised and semi-supervised training tracks.

### 23 July 2026
Issue: Completed method stages were reviewed; further extension was needed to improve performance beyond the frozen reference configuration.  
Action: Extend the framework with additional adaptive scoring and merge strategies, including multi-condition and selective ensemble approaches, using regression-guarded promotion so that only non-regressing updates are retained. Prepare comparative metrics and findings for the next review.

### 28 July 2026
Issue: The latest adaptive scoring and guarded-merge experiments were approved; formal documentation had not yet been prioritised.  
Action: Begin formal project documentation. Organise methodology, results, figures, and tables into a coherent draft, with emphasis on clarity and consistency in presenting methods and findings.
