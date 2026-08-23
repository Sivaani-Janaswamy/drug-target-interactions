# Work_Division.md — DTI-ML Team (2 Members)

Use real names where marked `[Name]`. This version keeps the same project scope but combines the original 3-person responsibilities into 2 roles.

## Progress Checklist

### Member 1
- [x] Repo scaffold setup complete
- [x] Shared environment file created
- [x] Initial project documentation finished
- [x] Initial app skeleton created
- [ ] Download + clean KIBA dataset
- [ ] Build split files
- [ ] Drug feature extraction pipeline
- [ ] Protein feature extraction pipeline
- [ ] Model training and tuning

### Member 2
- [x] Initial evaluation scaffold planned
- [x] App skeleton ready for wiring
- [ ] Metric validation and SHAP analysis
- [ ] Model integration into app
- [ ] App final polish and deployment
- [ ] Final paper compilation and presentation

---

## Member 1 — Data, Feature Engineering & Model Training Lead

**Owns:** dataset cleaning, splits, drug-side features, protein-side features, model training, tuning, and 1/2 of the paper.

| Week | Task | Works With |
|---|---|---|
| 1 | Repo scaffold setup, literature skim | Member 2 |
| 1 | Download + clean KIBA, EDA | Solo |
| 1 | Build 3 split versions (random, cold-drug, cold-protein) | Solo |
| 1 | Drug feature pipeline: RDKit fingerprints + descriptors | Solo |
| 1 | Protein feature pipeline: AAC/CTD/PseAAC via propy3 | Solo |
| 1 | Set up training scaffolding (scikit-learn, XGBoost, LightGBM) | Solo |
| 2 | Optional ESM-2 embedding extraction (frozen, cached) | Solo |
| 2 | Train baseline versions of RF, XGBoost/LightGBM, SVR, GPR on random split | Solo |
| 2 | Hyperparameter tuning pass 1 (all 4 models) | Solo |
| 2 | Feature ablation study across drug/protein/combined variants | Member 2 |
| 3 | Train/evaluate all 4 algorithms on cold-drug and cold-protein splits | Solo |
| 3 | Build the full 12-cell results matrix | Member 2 |
| 4 | Paper: Abstract, Introduction, Related Work, Methodology | Member 2 |
| 4 | Final paper read-through/edit pass | Member 2 |
| 4 | Presentation rehearsal | Member 2 |

**Deliverables owned:** cleaned dataset, split files, drug feature matrix, protein feature matrix, trained/tuned models, feature ablation results, paper sections (Abstract, Introduction, Related Work, Methodology).

---

## Member 2 — Evaluation, Interpretability, Web App & Paper Lead

**Owns:** evaluation metrics, SHAP analysis, app development, paper compilation, demo materials, and 1/2 of the paper.

| Week | Task | Works With |
|---|---|---|
| 1 | Repo scaffold setup, literature skim | Member 1 |
| 1 | Evaluation metric scaffolding (MSE, RMSE, CI, Pearson r) | Solo |
| 1 | Web app skeleton (page layout, no model wired in) | Solo |
| 2 | Validate evaluation pipeline against baseline models | Member 1 |
| 2 | Wire in the first trained model to the app | Solo |
| 2 | Support feature ablation analysis and result logging | Member 1 |
| 3 | SHAP analysis on 3–5 showcase predictions | Solo |
| 3 | Pharmacophore sanity-check on SHAP examples | Member 1 |
| 3 | Web app: model selector, SHAP overlay, GPR uncertainty display | Solo |
| 3 | Compile full 12-cell results matrix from model outputs | Member 1 |
| 4 | Finish web app (About page, gallery page, reference-drug comparison), deploy | Solo |
| 4 | Paper: Experimental Setup, Results & Discussion, Conclusion & Future Work, References | Member 1 |
| 4 | Record demo video/screenshots backup | Solo |
| 4 | Final compiled paper and presentation rehearsal | Member 1 |

**Deliverables owned:** evaluation pipeline, SHAP visualizations, complete web app, results tables, final paper compilation, demo backup materials.

---

## Shared Responsibilities (Both Members)

- Daily async check-in; weekly sync at the end of each week
- Final paper read-through and edit pass (Week 4)
- Presentation rehearsal and speaking-section assignment (Week 4)
- Immediate flagging of blockers whenever any pipeline slips behind schedule
- Shared ownership of the final app and submission-ready paper

## Handoff Points (Critical Dependencies)

| From | To | What | By When |
|---|---|---|---|
| Member 1 | Member 2 | Cleaned datasets, split files, feature matrices | End of Week 1 |
| Member 1 | Member 2 | Baseline trained model checkpoints and tuning results | Mid Week 2 |
| Member 1 | Member 2 | Full 12-cell result matrix and ablation outputs | End of Week 3 |
| Both | Both | Final result tables, SHAP figures, methodology notes for paper completion | Throughout Week 4 |

## Notes for a 2-Person Team

- The workload is intentionally distributed so one member owns data/model pipeline tasks while the other owns evaluation/app/paper flows.
- Both roles still collaborate heavily during model evaluation and final writing.
- If one member is blocked, the other should continue with downstream tasks instead of waiting, to minimize schedule risk.
