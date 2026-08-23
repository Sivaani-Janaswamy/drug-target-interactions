# Work_Division.md — DTI-ML Team (2 Members)

This document defines the project ownership model for a 2-person workflow. It separates technical execution from evaluation and final delivery while keeping the project integrated.

## Progress Checklist

### Member 1
- [x] Repo scaffold setup complete
- [x] Shared environment file created
- [x] Initial project documentation finished
- [x] Initial app skeleton created
- [ ] Download and validate KIBA dataset
- [ ] Run EDA and confirm dataset statistics
- [ ] Build split files
- [ ] Generate drug and protein features
- [ ] Train and tune models

### Member 2
- [x] Initial evaluation scaffold planned
- [x] App skeleton ready for wiring
- [ ] Validate metrics pipeline
- [ ] Integrate model output into app
- [ ] Complete SHAP and interpretability analysis
- [ ] Final app polish and deployment
- [ ] Final paper compilation and presentation prep

---

## Team Operating Model

### Member 1 — Data, Feature Engineering, and Model Training Lead
**Primary focus:** dataset engineering, feature generation, and model development

**Owned work**
- download and clean KIBA dataset
- exploratory data analysis
- build split files for random, cold-drug, and cold-protein scenarios
- generate RDKit drug fingerprints and descriptors
- generate protein features using AAC, CTD, and PseAAC
- set up model training and hyperparameter tuning
- produce model checkpoints and result tables

**Deliverables**
- cleaned dataset
- split files
- feature matrices
- trained and tuned regression models
- model results summary

**Review requirement**
- final model outputs must be checked with Member 2 before being used in the final paper or app

---

### Member 2 — Evaluation, Interpretability, Web App, and Final Delivery Lead
**Primary focus:** evaluation quality, interpretability, app development, and final documentation

**Owned work**
- metric scaffolding (MSE, RMSE, CI, Pearson r)
- model evaluation pipeline validation
- SHAP analysis and example interpretation
- app design and model integration
- final app polish and deployment
- final paper compilation and presentation prep

**Deliverables**
- evaluation pipeline
- SHAP figures and notes
- working web app
- final paper draft
- demo and presentation materials

**Review requirement**
- app and paper outputs must be checked against the model results before final submission

---

## Shared Responsibilities

Both members share responsibility for:
- weekly progress review
- blocker flagging when a dependency slips
- final paper review and approval
- final app check before presentation
- final presentation rehearsal

---

## Phase Ownership Matrix

| Phase | Primary Owner | Supporting Owner | Outcome |
|---|---|---|---|
| Phase 1 — Foundation and data acquisition | Member 1 | Member 2 | project ready for data work |
| Phase 2 — Data cleaning and splits | Member 1 | Member 2 | reproducible train/test design |
| Phase 3 — Feature engineering | Member 1 | Member 2 | usable feature sets |
| Phase 4 — Model training and tuning | Member 1 | Member 2 | trained models and result logs |
| Phase 5 — Evaluation and SHAP | Member 2 | Member 1 | final metrics and interpretation |
| Phase 6 — App and paper finalization | Member 2 | Member 1 | working app + submission-ready paper |

---

## Critical Dependencies

| Dependency | Depends On | Owner |
|---|---|---|
| model training can start | cleaned dataset + split files + feature matrices | Member 1 |
| evaluation results can be produced | trained model checkpoints | Member 1 |
| app can be finalized | validated model outputs | Member 2 |
| paper can be completed | final metrics, figures, and app evidence | Both |

---

## Working Rule

- Member 1 is responsible for the technical pipeline.
- Member 2 is responsible for interpretability, validation, and final product delivery.
- If one person is blocked, the other should continue with downstream work without waiting, while still recording the dependency clearly.
