# Project Spec: DTI-ML
### Classical Machine Learning for Kinase Drug–Target Affinity Prediction with Cold-Split Evaluation and Interpretable Feature Attribution

**Team size:** 3 | **Timeline:** 4 weeks | **Compute:** College lab CPU/GPU (GPU not required) | **Deliverables:** Working web app + IEEE-format paper (college evaluation)

---

## 1. Problem Statement

Predict the binding affinity between small-molecule drugs and kinase protein targets using classical machine learning algorithms, with two goals beyond a standard DTI regressor:

1. Evaluate under a **cold-split protocol** (no drug or protein overlap between train/test) to give an honest, non-inflated performance estimate — most published DTI models are evaluated with random splits that leak similarity information, and classical similarity-based ML methods (e.g., KronRLS-style approaches) are especially prone to this leakage.
2. Provide **interpretability** via SHAP feature attribution — show which molecular substructures and which protein composition features drove each prediction.

## 2. Why Kinases / This Domain

Kinases are one of the most drug-relevant protein families (cancer, inflammatory disease). Using the **KIBA** dataset keeps scope realistic for 4 weeks: it's public, pre-cleaned, widely benchmarked (used in DeepDTA, GraphDTA, KronRLS, SimBoost papers), and domain-specific enough to justify "specific application area" without requiring raw BindingDB curation.

- Primary dataset: [KIBA](https://github.com/hkmztrk/DeepDTA/tree/master/data) (~2,100 drugs × ~229 kinases, ~118K interaction pairs with KIBA affinity scores)
- Fallback/comparison dataset: Davis (smaller, ~30K pairs — useful if full KIBA feature extraction is too slow)

## 3. Novelty / Contributions (for the paper's "contribution" section)

1. **Systematic classical-ML algorithm comparison**: Random Forest, Gradient Boosting (XGBoost/LightGBM), Support Vector Regression, and Gaussian Process Regression, all evaluated on identical engineered features — most classical DTI papers report only one algorithm.
2. **Cold-split evaluation**: explicit unseen-drug and unseen-protein test splits reported alongside the standard random split, to quantify the generalization gap for each algorithm — classical similarity-based methods are known to be vulnerable here, and this is rarely shown explicitly in one place.
3. **Feature ablation study**: fingerprint-only vs. descriptor-only vs. combined vs. combined + frozen protein-language-model (ESM-2) embedding-as-feature, to isolate which engineered features actually drive performance and whether a modern PLM embedding helps a classical model generalize better under distribution shift.
4. **SHAP-based interpretability**: per-prediction feature attribution mapped back to concrete fingerprint bits (substructures) and protein composition statistics — more human-readable than attention weights, and validated qualitatively against known kinase-inhibitor pharmacophores (e.g., hinge-region binding motifs).
5. **Affinity regression** (continuous KIBA score) rather than binary interaction classification — harder and more clinically meaningful.
6. **Deployable demo**: interactive web app for real-time prediction + visualization, not just a notebook.

## 4. System Architecture

```
 Drug (SMILES)                         Protein (sequence)
      │                                        │
      ▼                                        ▼
 RDKit → Morgan/ECFP fingerprints      Feature extraction:
 + physicochemical descriptors         - Amino Acid Composition (AAC)
 (MW, LogP, TPSA, H-bond donors/       - CTD (Composition/Transition/
 acceptors, rotatable bonds, etc.)       Distribution) descriptors
      │                                - Pseudo Amino Acid Composition (PseAAC)
      │                                - Optional feature: frozen ESM-2
      │                                  embedding (mean-pooled), used as a
      │                                  fixed input vector — not trained
      │                                        │
      └──────────► Concatenated feature vector ◄──┘
                              │
                              ▼
              Classical ML regressor (trained + compared):
              - Random Forest Regressor
              - Gradient Boosting (XGBoost / LightGBM)
              - Support Vector Regression (RBF kernel)
              - Gaussian Process Regression (bonus: native uncertainty)
                              │
                              ▼
                  Predicted binding affinity (KIBA score)
                              │
                              ▼
              SHAP values → feature importance → substructure /
              protein-region highlight
```

## 5. Tech Stack

| Layer | Tool |
|---|---|
| Drug featurization | RDKit (Morgan/ECFP fingerprints, physicochemical descriptors) |
| Protein featurization | `propy3` / `iFeature` (AAC, CTD, PseAAC); optional `fair-esm` or HuggingFace `facebook/esm2_t12_35M` for frozen embeddings |
| Models | scikit-learn (Random Forest, SVR, Gaussian Process), XGBoost, LightGBM |
| Interpretability | SHAP |
| Experiment tracking | Weights & Biases (free tier) or plain CSV logs |
| Web app | React frontend + FastAPI backend |
| Paper | Overleaf, IEEE conference template |

## 6. Team Roles (3 members)

**Member A — Data & Drug Feature Pipeline**
- Download/clean KIBA, build cold-split (unseen-drug, unseen-protein, random baseline)
- SMILES → fingerprint + descriptor pipeline (RDKit)
- Feature ablation experiment design (drug side)

**Member B — Protein Feature Pipeline & Model Training**
- Protein sequence → AAC/CTD/PseAAC feature pipeline; optional ESM-2 embedding extraction (cached to disk)
- Train and tune all four classical ML algorithms (RF, XGBoost/LightGBM, SVR, GPR)
- Hyperparameter tuning (grid/random search), checkpointing best models

**Member C — Evaluation, Interpretability, Web App, Paper Lead**
- Evaluation metrics + cold-split result tables across all algorithms
- SHAP interpretability analysis + pharmacophore sanity-check on 3–5 examples
- React/FastAPI demo app (SMILES + protein sequence in → affinity + SHAP visualization out)
- IEEE paper drafting lead (compiles all members' results into paper sections; writing still split evenly — see plan.md)

## 7. Evaluation Metrics

- MSE, RMSE (primary — affinity regression)
- Concordance Index (CI) — standard in DTI literature, comparable to published papers (KronRLS, SimBoost, DeepDTA)
- Pearson correlation coefficient
- Reported **separately** for: random split / cold-drug split / cold-protein split, **for each of the 4 algorithms** (12-cell results matrix)

## 8. Web App Spec

**Input:** SMILES string (or dropdown of example drugs) + protein sequence (or dropdown of example kinases) + model selector (RF/XGBoost/SVR/GPR)
**Output:**
- Predicted affinity score (with plain-language interpretation, e.g., "strong/weak predicted binding")
- Drug structure rendered (RDKit → image)
- SHAP-based importance highlight overlaid on the drug structure / listed as top contributing features
- Prediction uncertainty (if GPR selected)
- Optional: compare against 2–3 known reference drugs for context

**Pages:** Home/predict page, "About the model" page (algorithm comparison table, for demo credibility), example gallery page

## 9. IEEE Paper Structure

1. Abstract (150–250 words)
2. Introduction (motivation, problem statement, contributions list)
3. Related Work (KronRLS, SimBoost, DeepDTA, GraphDTA — positioning classical ML vs. deep learning approaches)
4. Methodology (architecture diagram from Section 4, feature engineering details, dataset, split protocol)
5. Experimental Setup (hyperparameters per algorithm, hardware, training details)
6. Results & Discussion (12-cell metrics matrix, cold-split comparison, feature ablation results, SHAP visualization examples)
7. Conclusion & Future Work
8. References

## 10. Risks & Fallbacks

| Risk | Fallback |
|---|---|
| Protein feature extraction (CTD/PseAAC) slow on full KIBA | Subsample proteins or use `propy3` batch mode; cache features to disk once |
| GPR too slow on full dataset (O(n³) scaling) | Subsample to ~5–10K pairs for GPR only; keep RF/XGBoost/SVR on full data |
| Full KIBA too large to train in time | Subsample to ~20–30K pairs, note it as a limitation in the paper |
| Web app running behind | Keep the API boundary small and use the approved React mockup as the implementation base |
| SHAP computation slow for tree ensembles on full test set | Use `TreeExplainer` (fast, exact for RF/XGBoost) and compute SHAP only on the 3–5 showcase examples + a small validation subset |

## 11. Planned: Chatbot Feature

A future add-on will embed a scoped Q&A chatbot (Google Gemini Flash free tier) in the
web app so users can ask plain-language questions about the current prediction (score
meaning, cold-split rationale, SHAP interpretation) without leaving the page. It's
grounded to the current prediction's context, guarded against off-topic/medical-advice
questions, and rate-limited to protect the free API quota. See
[CHATBOT_SPEC.md](CHATBOT_SPEC.md) for the full architecture and contract — not yet
implemented.

## 12. Deliverables Checklist

- [ ] Cleaned dataset + 3 split versions (random, cold-drug, cold-protein)
- [ ] Drug feature pipeline (fingerprints + descriptors)
- [ ] Protein feature pipeline (AAC/CTD/PseAAC + optional ESM-2 embedding)
- [ ] 4 trained classical ML models + saved checkpoints
- [ ] Feature ablation results table
- [ ] Results table across all splits, metrics, and algorithms
- [ ] SHAP interpretability visualizations (min. 3 examples)
- [ ] Working React/FastAPI web app (deployed locally or on suitable free hosting)
- [ ] IEEE-format paper draft
- [ ] Presentation deck / demo script
