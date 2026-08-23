# Plan.md — DTI-ML: 4-Week Execution Plan

Team: 3 members (A, B, C — see work_division.md for names/roles mapping)
Goal: working web app + IEEE-format paper draft by end of Week 4

---

## Week 1 — Setup, Data, Feature Foundations

**Goals:** environment ready, data cleaned, splits built, both feature pipelines producing output on a sample.

| Day | Task | Owner |
|---|---|---|
| 1 | Repo scaffold (folders: `data/`, `features/`, `models/`, `app/`, `paper/`), shared environment file, literature skim (KronRLS, SimBoost, DeepDTA, GraphDTA — for Related Work) | All |
| 1–2 | Download KIBA (+ Davis as fallback), initial EDA (pair counts, score distribution, drug/protein counts) | A |
| 2–3 | Build 3 split versions: random, cold-drug (unseen drug in test), cold-protein (unseen protein in test); save as reusable split files | A |
| 2–4 | Drug feature pipeline: RDKit Morgan/ECFP fingerprints + physicochemical descriptors, run on full drug list | A |
| 2–4 | Protein feature pipeline: AAC/CTD/PseAAC via `propy3`, run on full protein list; scope out ESM-2 embedding extraction as optional add-on | B |
| 3–5 | Set up scikit-learn/XGBoost/LightGBM training scaffolding (data loaders reading feature + split files) | B |
| 4–5 | Set up SHAP + evaluation metric scaffolding (MSE, RMSE, CI, Pearson r) so it's ready to run the moment models exist | C |
| 5 | Web app skeleton (Streamlit page layout, no model wired in yet) | C |

**End-of-week deliverable:** cleaned dataset + 3 split files, drug + protein feature matrices saved to disk, repo scaffold, app skeleton.

---

## Week 2 — Model Training & First Results

**Goals:** all 4 algorithms trained on random split at least once; feature pipeline fully validated.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Train baseline versions of all 4 models (RF, XGBoost/LightGBM, SVR, GPR) on random split — get first numbers, even if unoptimized | B |
| 1–2 | Optional ESM-2 embedding extraction (frozen, batch inference, cached to disk) for feature ablation later | A |
| 2–3 | Hyperparameter tuning pass 1 (grid/random search) on random split for all 4 models | B |
| 3–4 | Run evaluation scaffolding on Week 2 models — confirm MSE/RMSE/CI/Pearson r pipeline works end-to-end | C |
| 3–5 | Feature ablation experiment: fingerprint-only vs. descriptor-only vs. combined vs. combined+ESM-2, on random split, using best-so-far algorithm | A + B |
| 4–5 | Web app: wire in one trained model (e.g., XGBoost) for a working input → prediction flow | C |
| 5 | Team sync: review first results table, decide if any algorithm needs to be dropped/replaced per Section 10 risks in spec.md | All |

**End-of-week deliverable:** first full results table (random split, all 4 algorithms), feature ablation results, working single-model prediction in the app.

---

## Week 3 — Cold-Split Evaluation, Interpretability, Full App

**Goals:** the paper's key result (cold-split comparison) is complete; SHAP visualizations done; app supports all models.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Train/evaluate all 4 algorithms on cold-drug and cold-protein splits (reuse tuned hyperparameters from Week 2) | B |
| 1–3 | Build the full 12-cell results matrix (4 algorithms × 3 splits × metrics); start drafting Results section notes | A + B |
| 2–4 | SHAP analysis: run `TreeExplainer`/`KernelExplainer` on 3–5 showcase predictions, generate substructure/feature highlight plots | C |
| 3–4 | Pharmacophore sanity-check: manually compare SHAP-highlighted substructures against known kinase-inhibitor binding motifs for the showcase examples | A + C |
| 3–5 | Web app: add model selector (all 4 algorithms), SHAP visualization overlay, uncertainty display for GPR | C |
| 5 | Team sync: confirm all Section 11 (spec.md) deliverables except paper/app-polish are done | All |

**End-of-week deliverable:** full 12-cell results table, feature ablation table, SHAP visualizations with pharmacophore validation notes, app with model selector + interpretability view.

---

## Week 4 — Web App Polish, Paper, Presentation

**Goals:** app deployed and demo-ready; paper complete; presentation rehearsed.

| Day | Task | Owner |
|---|---|---|
| 1–2 | Finish web app: "About the model" page, example gallery page, reference-drug comparison feature; deploy (Streamlit Cloud or local) | C |
| 1–2 | Paper: Abstract, Introduction, Related Work | A |
| 1–2 | Paper: Methodology (architecture, feature engineering, split protocol) | B |
| 2–3 | Paper: Experimental Setup, Results & Discussion (tables + SHAP figures) | C (compiles) + A + B (provide content) |
| 3–4 | Paper: Conclusion & Future Work, References; full read-through and edit pass | All |
| 4 | Record demo video/screenshots as backup in case live demo fails | C |
| 4–5 | Rehearse presentation, assign speaking sections | All |
| 5 | Final buffer day: fix any last-minute app bugs, finalize paper submission version | All |

**End-of-week deliverable:** working deployed app, submission-ready IEEE paper, presentation deck + rehearsed demo.

---

## Cross-Cutting Notes

- **Sync cadence:** short check-in at the end of each day (async message is fine), full team sync every Friday (end of week) per the table above.
- **Blocking dependencies:** Week 2 model training (B) depends on Week 1 feature pipelines (A) — if A's features slip past Day 4 of Week 1, flag immediately so B can start with a partial feature set rather than waiting.
- **Paper writing is split evenly across all 3 members regardless of technical role**, per spec.md Section 6 — Week 4 paper tasks above reflect that split.
