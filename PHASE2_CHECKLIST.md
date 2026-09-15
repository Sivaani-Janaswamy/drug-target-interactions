# DTI-ML — Phase 2 Research Strengthening Checklist

**Purpose:** This is the authoritative task list for Phase 2 of the DTI-ML project (KIBA kinase drug–target affinity prediction). Each task below has a research question, exact scope, required deliverables, and pass/fail acceptance criteria. A coding agent implementing any task must:

1. Read the task's **Scope** and **Do NOT** sections before writing code.
2. Implement only what's listed under **Deliverables**.
3. Run the checks under **Acceptance Criteria** and confirm every box passes before marking the task `[x]`.
4. Never mark a task complete if any acceptance check fails, is skipped, or produces an error that was silently caught.

Baseline context: Drug features = 1,030 dims. Protein features = 197 dims (AAC=20, CTD=147, PseAAC=30). Existing models: RF, XGBoost, SVR, GPR. Dataset: KIBA. Evaluation: cold-split protocol, 3 split settings, seed=42 currently.

---

## Foundation (already complete — do not re-verify unless regression suspected)

- [x] Real KIBA data pipeline
- [x] Leakage-safe cold splits
- [x] 4 ML models (RF, XGBoost, SVR, GPR)
- [x] 3 evaluation settings, 12 experiments total
- [x] Reproducibility (fixed seed=42 runs)
- [x] SHAP explainability
- [x] Chatbot feature
- [x] Documentation
- [x] Manuscript synchronized with foundation results

---

## 2.1 — Simple Baselines 🟢

**Research question:** Do RF/XGBoost/SVR/GPR actually beat trivial predictors?

**Scope / Deliverables:**
- Implement at minimum: (a) mean/constant prediction, (b) linear regression. Add a basic non-linear baseline (e.g. k-NN) only if it's trivial to add.
- Run each baseline through the **same** cold-split protocol and same 3 evaluation settings as existing models (no shortcuts).
- Produce one results table: baseline vs RF/XGBoost/SVR/GPR, same metrics already used in the project (e.g. RMSE, CI, Pearson).
- Save table as a file (CSV + rendered in report/notebook), not just printed to console.

**Do NOT:**
- Do not tune the baselines. They must stay "simple" by design.
- Do not touch the existing model training code.

**Acceptance Criteria (all must pass):**
- [ ] Baseline models trained on identical train/test splits as the main models (verify by split hash/seed, not just "same seed number").
- [ ] Same metric functions used for baselines and main models (no separate metric implementation).
- [ ] Results table exists as a saved artifact and includes all 3 split settings.
- [ ] Table shows baselines clearly underperforming (or, if not, this is flagged explicitly — do not hide unexpected results).
- [ ] No modification to existing model files.

---

## 2.2 — Feature Ablation 🟢🟡

**Research question:** Do drug/protein feature groups (AAC, CTD, PseAAC, drug fingerprint) meaningfully contribute?

**Scope / Deliverables:**
- Define a small, fixed set of ablation configurations (do not exceed ~5–6):
  - Full (Drug 1030 + AAC 20 + CTD 147 + PseAAC 30)
  - No PseAAC
  - No CTD
  - No AAC
  - No Drug features (protein-only, as a sanity floor)
- Retrain the **best-performing existing model** (not all 4) on each ablation config, same split protocol.
- Produce an ablation table (config × metrics) and one plot showing performance delta vs Full.

**Do NOT:**
- Do not create more than ~6 ablation configs "just to be thorough."
- Do not change feature extraction code — only change which columns are selected before training.

**Acceptance Criteria (all must pass):**
- [ ] Feature selection is done via a config flag/column mask, not by editing feature extraction scripts.
- [ ] Each ablation config's feature count is logged and matches expectation (e.g. "No PseAAC" = 1030+20+147 = 1197 dims — verify programmatically, not by eyeballing).
- [ ] All ablations evaluated on identical splits as Full config.
- [ ] Ablation table + plot saved as artifacts.
- [ ] Written interpretation (1–2 sentences per config) included in report, not just raw numbers.

---

## 2.3 — Pairwise Interaction Features 🟡

**Research question:** Does explicit drug–protein interaction representation (vs. plain concatenation) improve prediction?

**Scope / Deliverables:**
- Implement interaction features as a separate feature-engineering step, controlled and small: element-wise product, absolute difference (and optionally 1–2 more well-justified statistics — no more).
- Compare: baseline concatenation `[drug || protein]` vs `[drug || protein || interaction features]`.
- Use same model(s) and same evaluation protocol for both.

**Do NOT:**
- Do not let the interaction feature space explode (e.g. don't cross every drug dim with every protein dim — that's 1030×197 ≈ 200K dims, which is out of scope). Element-wise ops require equal dimensionality — clarify/resolve the dimension mismatch (1030 vs 197) explicitly before implementing (e.g. via a projection or restricting interaction to a shared reduced space), and document the chosen approach.
- Do not silently drop this dimension-mismatch issue — it must be resolved and documented, not worked around invisibly.

**Acceptance Criteria (all must pass):**
- [ ] The dimension-mismatch problem (1030 drug dims vs 197 protein dims) is explicitly addressed in code comments/docs, with the chosen resolution stated.
- [ ] Interaction feature dimensionality is fixed and logged.
- [ ] Concatenation-only vs interaction-enhanced runs use identical splits, seeds, and model hyperparameters — only the feature set differs.
- [ ] Results table shows both configs side by side with metric deltas.
- [ ] No unbounded feature explosion (verify final feature count is reasonable, e.g. same order of magnitude as original 1227 dims).

---

## 2.4 — LightGBM / CatBoost 🟢🟡

**Research question:** Is XGBoost uniquely strong among gradient boosting methods, or are LightGBM/CatBoost competitive?

**Scope / Deliverables:**
- Add LightGBM and CatBoost as two new model options, following the exact same interface/wrapper pattern already used for RF/XGBoost/SVR/GPR.
- Run both across all 3 existing evaluation settings.
- Extend the existing results table to include: RF, XGBoost, LightGBM, CatBoost, SVR, GPR (+ baselines from 2.1 if available).

**Do NOT:**
- Do not hyperparameter-tune LightGBM/CatBoost more extensively than the existing models were tuned — keep tuning effort comparable/fair across all models.
- Do not create separate/duplicate evaluation code paths — reuse the existing pipeline.

**Acceptance Criteria (all must pass):**
- [ ] LightGBM and CatBoost integrate into the existing model factory/interface without duplicating the training loop.
- [ ] Both new models produce results for all 3 split settings.
- [ ] Combined comparison table (6 models, or 6+baselines) saved as one artifact.
- [ ] Tuning budget/hyperparameter search space documented and comparable across all 6 models (not just default-vs-tuned mismatch).
- [ ] No breakage of existing RF/XGBoost/SVR/GPR results — rerun and confirm they match previously recorded values (regression check).

---

## 2.5 — Repeated Splits + Mean ± SD / CI 🟡🟠

**Research question:** Are results stable across different splits/seeds, or seed-dependent luck?

**Scope / Deliverables:**
- **Before coding:** design and document the experiment matrix explicitly. Do not blindly repeat everything.
  - Recommended default: models finalized after 2.4 × 3 split types × 5 seeds. Confirm actual model count with the team before running (this determines total experiment count — calculate and log it).
- Implement seed-controlled repetition of the existing training/eval pipeline (only the seed changes between runs — same code path).
- Aggregate results as mean ± SD (and/or 95% CI) per model per split type.
- Produce a comparison table/plot that makes stability visible (e.g. error bars), not just averages.

**Do NOT:**
- Do not silently reduce scope (e.g. dropping split types or models) without updating the documented experiment matrix first.
- Do not report only the mean and discard the SD — instability is a valid and important finding.

**Acceptance Criteria (all must pass):**
- [ ] Experiment matrix (models × splits × seeds = total runs) is written down and matches what was actually executed — verify count of result files/rows equals the planned total exactly.
- [ ] Each repeated run uses a different seed but otherwise identical config (verify via logged configs, not assumption).
- [ ] Aggregated table reports mean ± SD (or CI) for every model/split combination — no missing cells.
- [ ] At least one model shows a case where SD materially affects interpretation (e.g. two models' means overlap within SD) — explicitly called out in the written summary, or explicitly confirmed absent.
- [ ] Runtime/compute cost logged so future repeats can be estimated.

---

## 2.6 — ESM-2 / ProtT5 Protein Embeddings 🟠🟠

**Research question:** Do pretrained protein language-model embeddings outperform handcrafted descriptors (AAC/CTD/PseAAC)?

**Scope / Deliverables — plan before implementing:**
- Decide ESM-2 **or** ProtT5 (pick one to start; document why).
- Define and document: embedding extraction method, pooling strategy (mean pooling vs CLS-token equivalent), output dimensionality, storage format, and how it combines with the 1,030-dim drug representation.
- Decide whether dimensionality reduction (PCA/similar) is needed for fair comparison, and document the decision.
- Implement as an **alternative** protein feature branch — do not delete/replace the existing 197-dim descriptor pipeline. Both must remain runnable and comparable.
- Run the same model(s)/eval protocol with ESM-2/ProtT5 features swapped in for AAC/CTD/PseAAC.

**Do NOT:**
- Do not bolt this onto the pipeline without the planning step above being written down first.
- Do not remove or overwrite the existing handcrafted-feature pipeline — this is a comparison, not a replacement.
- Do not skip documenting compute/storage requirements (embedding files can be large — confirm feasibility before running full dataset).

**Acceptance Criteria (all must pass):**
- [ ] A short design doc/section exists covering: model choice, pooling, dimensionality, storage, and combination strategy — written **before** the implementation was merged.
- [ ] Existing AAC/CTD/PseAAC pipeline still runs unmodified and produces the same results as before (regression check).
- [ ] Embedding extraction is reproducible (same protein sequence → same embedding, verified on a sample).
- [ ] Final comparison table: handcrafted features vs pretrained embeddings, same models, same splits.
- [ ] Compute time and storage size for embeddings logged and reported.

---

## 2.7 — Davis External Validation 🟠🟠

**Research question:** Does the trained approach generalize beyond KIBA?

**Scope / Deliverables — plan before implementing:**
- Explicitly document how Davis and KIBA differ in: drug identifiers, protein identifiers, affinity scale/units, and preprocessing — resolve each before running anything.
- Train/finalize model on KIBA only (frozen — no peeking at Davis during training/tuning).
- Preprocess Davis independently using the documented mapping, producing feature vectors in the exact same format/dimensionality as the KIBA-trained model expects.
- Evaluate the frozen KIBA model on Davis; report metrics with the same functions used elsewhere.

**Do NOT:**
- Do not directly feed downloaded Davis data into the existing model without addressing identifier/scale/preprocessing mismatches first — this is the single most likely source of a silently broken result.
- Do not retrain or fine-tune on Davis and call it "external validation" — that would defeat the purpose.

**Acceptance Criteria (all must pass):**
- [ ] Written mapping/reconciliation of drug IDs, protein IDs, and affinity scale between KIBA and Davis exists and is used in code (not assumed identical).
- [ ] Davis feature vectors verified to have identical shape/ordering to what the KIBA-trained model expects (assert this programmatically, don't assume).
- [ ] Model weights used for Davis evaluation are confirmed identical to the frozen KIBA-trained model (no retraining step run in between).
- [ ] Affinity values compared on a consistent scale (converted if needed, with conversion documented).
- [ ] Final external-validation metrics reported alongside KIBA in-domain metrics for direct comparison.

---

## 2.8 — Uncertainty Calibration 🟠🟠

**Research question:** Does model confidence correspond to actual prediction error?

**Scope / Deliverables:**
- Use GPR's native predictive variance as the primary uncertainty source (it's already in the pipeline).
- If extending uncertainty to other models (RF/XGBoost/etc.), document the method chosen (e.g. quantile regression, ensemble variance) rather than assuming one exists by default.
- Produce: calibration/reliability plot, and an analysis of error vs. predicted uncertainty (e.g. binned error vs uncertainty, or coverage of prediction intervals).

**Do NOT:**
- Do not claim calibration results for models that don't natively produce uncertainty without first documenting how uncertainty was derived for them.
- Do not run this before 2.1–2.5 are stable — this should build on frozen, validated model results.

**Acceptance Criteria (all must pass):**
- [ ] Uncertainty source is explicitly stated per model (native for GPR; documented method for others).
- [ ] Reliability/calibration plot generated and saved.
- [ ] Quantitative check included (e.g. % of true values falling inside stated confidence interval — should roughly match the stated confidence level).
- [ ] Analysis explicitly states whether high-uncertainty predictions do or do not correlate with higher error (report the finding either way, not just if positive).

---

## 2.9 — Manuscript Update 🟢🟡

**Scope:** Do this **only after** all prior selected experiments are frozen — no more reruns after this starts.

**Deliverables:**
- Integrate new tables/figures from completed tasks into the manuscript.
- Update methods section to describe any new techniques used (ablation, interaction features, new models, embeddings, external validation, calibration — whichever were done).
- Update results/discussion/conclusion sections accordingly.

**Acceptance Criteria (all must pass):**
- [ ] All referenced tables/figures in the manuscript match the actual saved artifacts (no stale/placeholder numbers).
- [ ] No experiment referenced in the manuscript is missing from the results artifacts, and vice versa.
- [ ] Methods section accurately describes what was actually implemented (not the original plan, if scope changed).
- [ ] Full document reviewed end-to-end for consistency after edits.

---

## Explicitly Out of Scope for Phase 2

Do not implement unless separately requested and scoped:
- [ ] GNN-based models
- [ ] DeepTDA
- [ ] Multi-task learning
- [ ] Large-scale hyperparameter search
- [ ] Additional datasets beyond KIBA/Davis
- [ ] Vector databases
- [ ] Architecture changes not listed above

---

## Recommended Execution Order

1. **2.1 → 2.2 → 2.4** (low risk, independent, can be done in any order among themselves)
2. **2.3** (medium — resolve dimension mismatch first)
3. **2.5** (design experiment matrix before running)
4. **2.6, 2.7, 2.8** — each requires its own design/planning step before implementation; do not batch these into one large coding pass
5. **2.9** last, only after all included experiments are frozen

## Global Rule for the Coding Agent

For every task above: if any acceptance criterion cannot be verified programmatically (e.g. "results look reasonable"), do not mark the task complete — flag it back for human review instead of guessing.
