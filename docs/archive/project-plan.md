# Plan.md — DTI-ML: 4-Week Execution Plan

Team: 2 members (Member 1 and Member 2)
Goal: working web app + IEEE-format paper draft by end of Week 4

## Project Status Tracker

### Completed
- [x] Repo scaffold created
- [x] Python environment + dependency setup complete
- [x] Initial app skeleton created
- [x] Documentation structure prepared

### Not Started / Planned
- [ ] Download and validate KIBA dataset
- [ ] Run EDA on interaction counts and score distribution
- [ ] Build random, cold-drug, and cold-protein splits
- [ ] Generate drug features with RDKit
- [ ] Generate protein features with propy3
- [ ] Train baseline models on random split
- [ ] Tune and compare model performance
- [ ] Run evaluation metrics and SHAP analysis
- [ ] Connect model output to web app
- [ ] Finalize paper and presentation

---

## Phase Structure

### Phase 1 — Foundation and Data Acquisition
**Focus:** environment, data source, project setup, literature review

**Member responsibilities**
- Member 1: dataset sourcing, validation, and EDA
- Member 2: baseline app skeleton and evaluation scaffolding

**Deliverables**
- repository and folders ready
- dataset downloaded and validated
- initial project notes and literature references

**Exit criteria**
- dataset can be loaded in Python
- key statistics are understood
- project structure is ready for feature generation

---

### Phase 2 — Data Cleaning and Split Generation
**Focus:** train/test design and leakage-safe evaluation setup

**Member responsibilities**
- Member 1: clean the interaction table and construct splits
- Member 2: support with evaluation design and dataset format checks

**Deliverables**
- cleaned interaction data
- random split file
- cold-drug split file
- cold-protein split file

**Exit criteria**
- all three splits are reproducible
- no obvious leakage or formatting issues remain

---

### Phase 3 — Feature Engineering
**Focus:** convert molecules and proteins into usable ML features

**Member responsibilities**
- Member 1: RDKit drug features and protein feature extraction
- Member 2: validation of feature dimensions and compatibility with model inputs

**Deliverables**
- drug fingerprints and descriptors
- protein AAC / CTD / PseAAC features
- feature matrices saved to disk

**Exit criteria**
- feature matrices are created for all samples
- shapes and missing values are verified
- model-ready inputs are available

---

### Phase 4 — Model Training and Tuning
**Focus:** train and compare classical ML regressors

**Member responsibilities**
- Member 1: train and tune RF, XGBoost/LightGBM, SVR, GPR
- Member 2: evaluation pipeline and model result logging

**Deliverables**
- baseline model results
- tuned model checkpoints
- first comparison table

**Exit criteria**
- all 4 models have at least one working training run
- training/evaluation results are recorded

---

### Phase 5 — Evaluation, Interpretability, and App
**Focus:** cold-split performance, SHAP explanations, working app demo

**Member responsibilities**
- Member 1: run full split evaluation and ablation experiments
- Member 2: SHAP analysis, metrics validation, and app UI integration

**Deliverables**
- 12-cell results matrix
- feature ablation table
- SHAP visualizations
- app with prediction flow

**Exit criteria**
- models are evaluated across random, cold-drug, and cold-protein splits
- app predicts affinity input from user data
- interpretability outputs are visible and understandable

---

### Phase 6 — Paper, Presentation, and Final Delivery
**Focus:** final writing, polishing, and demo-ready delivery

**Member responsibilities**
- Member 1: methodology, dataset, model explanation, and paper sections
- Member 2: experimental results, SHAP section, web app, final compilation

**Deliverables**
- final IEEE-format paper draft
- final demo workflow
- presentation and rehearsal notes

**Exit criteria**
- paper is internally consistent and complete
- app is working and demo-ready
- team has a final presentation plan

---

## Week-by-Week Schedule

### Week 1 — Setup, Data, Feature Foundations
| Day | Task | Owner |
|---|---|---|
| 1 | Repo scaffold, environment setup, literature skim | Both |
| 1–2 | Download KIBA and validate data | Member 1 |
| 2–3 | Run EDA and confirm dataset statistics | Member 1 |
| 2–4 | Generate split files | Member 1 |
| 2–5 | Drug and protein feature extraction pipeline | Member 1 |
| 4–5 | Evaluation scaffolding and app skeleton | Member 2 |

**Goal:** dataset and feature pipeline ready for training.

### Week 2 — Model Training and First Results
| Day | Task | Owner |
|---|---|---|
| 1–2 | Train baseline models on random split | Member 1 |
| 2–3 | Hyperparameter tuning | Member 1 |
| 3–4 | Validate metrics pipeline | Member 2 |
| 3–5 | Feature ablation and initial results review | Both |
| 4–5 | Wire one trained model into the app | Member 2 |

**Goal:** first trained results and a usable demo model.

### Week 3 — Cold-Split Evaluation and Interpretability
| Day | Task | Owner |
|---|---|---|
| 1–2 | Evaluate models on cold-drug and cold-protein splits | Member 1 |
| 1–3 | Build full results matrix | Both |
| 2–4 | SHAP analysis and example validation | Member 2 |
| 3–5 | Add full app interaction and model selector | Member 2 |

**Goal:** complete results and interpretability outputs.

### Week 4 — Final App, Paper, and Presentation
| Day | Task | Owner |
|---|---|---|
| 1–2 | Final app polish and deployment | Member 2 |
| 1–2 | Paper: abstract, intro, method, related work | Member 1 |
| 2–3 | Paper: results, discussion, experimental setup | Member 2 |
| 3–4 | Final paper review and revision pass | Both |
| 4–5 | Presentation rehearsal and final checks | Both |

**Goal:** submission-ready app and paper.

---

## Dependencies and Blockers
- Model training cannot begin until the feature matrices are generated and validated.
- Evaluation and SHAP analysis depend on trained model checkpoints.
- App integration depends on model output and metrics being stable.
- Paper writing depends on final experiment results and figures.

## Working Rule
- Member 1 owns the data pipeline, feature engineering, and model training work.
- Member 2 owns the evaluation, interpretation, app, and paper compilation work.
- Both members review the final output before it is marked complete.
