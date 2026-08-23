# Plan.md — DTI-ML: 4-Week Execution Plan

Team: 2 members (Member 1 and Member 2 — see work_division.md for role mapping)
Goal: working web app + IEEE-format paper draft by end of Week 4

## Progress Tracker

### Completed
- [x] Repo scaffold created (`data/`, `features/`, `models/`, `app/`, `paper/`)
- [x] Shared environment file and Python dependency setup complete
- [x] Initial app skeleton created
- [x] Project documentation and folder structure finalized

### In Progress / Planned
- [ ] Download KIBA dataset and validate data format
- [ ] Run EDA and confirm dataset statistics
- [ ] Build random, cold-drug, and cold-protein splits
- [ ] Generate drug features with RDKit
- [ ] Generate protein features with propy3
- [ ] Train baseline models
- [ ] Tune models and compare performance
- [ ] Build SHAP interpretability and final app polish
- [ ] Write paper sections and final presentation

---

## Week 1 — Setup, Data, Feature Foundations

**Goals:** environment ready, data cleaned, splits built, both feature pipelines producing output on a sample.

| Day | Task | Owner |
|---|---|---|
| 1 | Repo scaffold (folders: `data/`, `features/`, `models/`, `app/`, `paper/`), shared environment file, literature skim (KronRLS, SimBoost, DeepDTA, GraphDTA — for Related Work) | Both |
| 1–2 | Download KIBA (+ Davis as fallback), initial EDA (pair counts, score distribution, drug/protein counts) | Member 1 |
| 2–3 | Build 3 split versions: random, cold-drug (unseen drug in test), cold-protein (unseen protein in test); save as reusable split files | Member 1 |
| 2–4 | Drug feature pipeline: RDKit Morgan/ECFP fingerprints + physicochemical descriptors, run on full drug list | Member 1 |
| 2–4 | Protein feature pipeline: AAC/CTD/PseAAC via `propy3`, run on full protein list; scope out ESM-2 embedding extraction as optional add-on | Member 1 |
| 3–5 | Set up scikit-learn/XGBoost/LightGBM training scaffolding (data loaders reading feature + split files) | Member 1 |
| 4–5 | Set up SHAP + evaluation metric scaffolding (MSE, RMSE, CI, Pearson r) so it's ready to run the moment models exist | Member 2 |
| 5 | Web app skeleton (Streamlit page layout, no model wired in yet) | Member 2 |

**End-of-week deliverable:** cleaned dataset + 3 split files, drug + protein feature matrices saved to disk, repo scaffold, app skeleton.

---

## Week 2 — Model Training & First Results

**Goals:** all 4 algorithms trained on random split at least once; feature pipeline fully validated.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Train baseline versions of all 4 models (RF, XGBoost/LightGBM, SVR, GPR) on random split — get first numbers, even if unoptimized | Member 1 |
| 1–2 | Optional ESM-2 embedding extraction (frozen, batch inference, cached to disk) for feature ablation later | Member 1 |
| 2–3 | Hyperparameter tuning pass 1 (grid/random search) on random split for all 4 models | Member 1 |
| 3–4 | Run evaluation scaffolding on Week 2 models — confirm MSE/RMSE/CI/Pearson r pipeline works end-to-end | Member 2 |
| 3–5 | Feature ablation experiment: fingerprint-only vs. descriptor-only vs. combined vs. combined+ESM-2, on random split, using best-so-far algorithm | Both |
| 4–5 | Web app: wire in one trained model (e.g., XGBoost) for a working input → prediction flow | Member 2 |
| 5 | Weekly sync: review first results table, decide if any algorithm needs to be dropped/replaced per Section 10 risks in spec.md | Both |

**End-of-week deliverable:** first full results table (random split, all 4 algorithms), feature ablation results, working single-model prediction in the app.

---

## Week 3 — Cold-Split Evaluation, Interpretability, Full App

**Goals:** the paper's key result (cold-split comparison) is complete; SHAP visualizations done; app supports all models.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Train/evaluate all 4 algorithms on cold-drug and cold-protein splits (reuse tuned hyperparameters from Week 2) | Member 1 |
| 1–3 | Build the full 12-cell results matrix (4 algorithms × 3 splits × metrics); start drafting Results section notes | Both |
| 2–4 | SHAP analysis: run `TreeExplainer`/`KernelExplainer` on 3–5 showcase predictions, generate substructure/feature highlight plots | Member 2 |
| 3–4 | Pharmacophore sanity-check: manually compare SHAP-highlighted substructures against known kinase-inhibitor binding motifs for the showcase examples | Both |
| 3–5 | Web app: add model selector (all 4 algorithms), SHAP visualization overlay, uncertainty display for GPR | Member 2 |
| 5 | Weekly sync: confirm all Section 11 (spec.md) deliverables except paper/app-polish are done | Both |

**End-of-week deliverable:** full 12-cell results table, feature ablation table, SHAP visualizations with pharmacophore validation notes, app with model selector + interpretability view.

---

## Week 4 — Web App Polish, Paper, Presentation

**Goals:** app deployed and demo-ready; paper complete; presentation rehearsed.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Finish web app: "About the model" page, example gallery page, reference-drug comparison feature; deploy (Streamlit Cloud or local) | Member 2 |
| 1–2 | Paper: Abstract, Introduction, Related Work | Member 1 |
| 1–2 | Paper: Methodology (architecture, feature engineering, split protocol) | Member 1 |
| 2–3 | Paper: Experimental Setup, Results & Discussion (tables + SHAP figures) | Member 2 (compiles) + Member 1 (provides content) |
| 3–4 | Paper: Conclusion & Future Work, References; full read-through and edit pass | Both |
| 4 | Record demo video/screenshots as backup in case live demo fails | Member 2 |
| 4–5 | Rehearse presentation, assign speaking sections | Both |
| 5 | Final buffer day: fix any last-minute app bugs, finalize paper submission version | Both |

**End-of-week deliverable:** working deployed app, submission-ready IEEE paper, presentation deck + rehearsed demo.

---

## Cross-Cutting Notes

- **Sync cadence:** short check-in at the end of each day (async message is fine), weekly review every Friday (end of week) per the table above.
- **Blocking dependencies:** Week 2 model training depends on Week 1 feature pipelines — if the feature pipeline slips past Day 4 of Week 1, flag it immediately so the next tasks can start with a partial feature set rather than waiting.
- **Paper writing is split across both members**, with the work distributed between data/model work and app/evaluation/paper compilation responsibilities.
