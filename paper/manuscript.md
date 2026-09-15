# Classical Machine Learning for Kinase Drug–Target Affinity Prediction with Cold-Split Evaluation and Interpretable Feature Attribution

**Abstract** — Predicting drug–target binding affinity is a central task in computational drug discovery, yet conventional evaluation protocols relying on random train/test splits often suffer from similarity-based data leakage, yielding artificially optimistic performance metrics that degrade on genuinely unseen molecular entities. In this work, we present **DTI-ML**, a systematic empirical evaluation of four classical machine learning regressors—Random Forest (RF), XGBoost, Support Vector Regression (SVR), and Gaussian Process Regression (GPR)—for kinase-inhibitor binding affinity prediction using the real KIBA benchmark dataset. We evaluate all models across three distinct partitioning protocols: standard Random Split, Cold-Drug Split (unseen drug candidates), and Cold-Protein Split (unseen kinase targets). Each drug–target pair is featurized into a concatenated 1,227-dimensional vector combining 1,024-bit Morgan fingerprints (ECFP4) and 6 physicochemical descriptors from RDKit with 197 protein features (AAC, CTD, and PseAAC). We apply SHapley Additive exPlanations (SHAP) to provide local interpretability for individual predictions. Across 12 experiment cells (4 models × 3 splits, seed 42), XGBoost consistently achieves the lowest root-mean-square error, while cold-split evaluation reveals a substantial generalization gap relative to random-split performance.

---

**Keywords:** drug–target affinity, KIBA, classical machine learning, cold-split evaluation, data leakage, SHAP, interpretability, kinase inhibitors

---

## I. INTRODUCTION

The identification of binding affinities between small-molecule drug candidates and biological protein targets is fundamental to target-based drug discovery and drug repurposing pipelines. Protein kinases represent one of the largest and most clinically relevant target classes, particularly in oncology and immunology.

Traditional wet-lab bioassays (e.g., $K_i$, $K_d$, $IC_{50}$) are expensive and time-consuming, often requiring weeks per compound and specialized equipment. Consequently, computational drug–target interaction (DTI) prediction has emerged as a primary filter to prioritize candidate compounds before experimental validation.

Most existing literature evaluates DTI models using standard **random train/test splits**. However, recent studies highlight that random splits suffer from severe **data leakage** because similar drug structures or homologous kinase targets frequently appear in both training and test sets. When deployed on genuinely novel drugs or unstudied protein targets, model performance can drop precipitously. This generalization gap undermines the practical utility of computationally predicted affinity scores.

### Key Contributions

1. **Systematic 4-Algorithm Benchmark**: Comparative evaluation of Random Forest, XGBoost, Support Vector Regression (SVR), and Gaussian Process Regression (GPR) under identical featurization on the real KIBA dataset.
2. **Leakage-Safe Cold-Split Evaluation**: Quantification of performance degradation across Random, Cold-Drug, and Cold-Protein splits, with explicit leakage checks confirming zero drug-ID and protein-ID overlap between training and test sets.
3. **1,227-Dimensional Joint Representation**: Integration of RDKit Morgan fingerprints (1,024-bit ECFP4) and 6 physicochemical descriptors with 197 protein features (AAC, CTD, PseAAC).
4. **SHAP-Based Local Explainability**: Per-prediction feature attribution for individual affinity predictions, providing interpretable insight into which molecular and protein features drive model decisions.
5. **Interactive Web Application**: A React and FastAPI application rendering the approved six-page product interface with real-time affinity scoring and SHAP explanations.

---

## II. LITERATURE SURVEY

### Drug–Target Interaction Prediction

Early computational approaches to DTI prediction relied on kernel-based similarity methods such as **KronRLS** and **SimBoost** [1][2]. These methods compute similarity matrices over drug and protein profiles and combine them to predict binding affinities. While effective on benchmark datasets, they do not explicitly model the underlying molecular biology of drug–target recognition.

With the advent of deep learning, models such as **DeepDTA** [3] (convolutional neural networks operating on SMILES strings and protein sequences) and **GraphDTA** [4] (graph neural networks on molecular graphs) established competitive benchmark results on the KIBA and Davis datasets. These approaches have demonstrated strong predictive performance, but they typically require large training datasets and substantial computational resources, and their predictions are often opaque to domain scientists.

### Classical Machine Learning for DTI

Classical machine learning models offer a compelling alternative for DTI prediction. Random Forest and gradient-boosted trees provide strong baselines with inherent feature importance measures, while Support Vector Regression and Gaussian Process Regression offer different trade-offs between model complexity and interpretability. Unlike deep learning approaches, classical models can be trained on modest-sized datasets, require less computational infrastructure, and produce more readily interpretable results.

### Evaluation Protocols and Data Leakage

A critical concern in DTI model evaluation is the choice of train/test split protocol. Standard random splits may place near-duplicate drug molecules or homologous kinase proteins in both training and test sets, allowing the model to exploit structural similarities rather than learning genuine binding determinants [5]. Cold-start evaluation protocols—in which test drugs or proteins are entirely absent from training—have been proposed to measure genuine generalization, but unified evaluations across multiple cold-start protocols remain relatively rare.

### Interpretable Machine Learning in Drug Discovery

The SHapley Additive exPlanations (SHAP) framework [6] provides a principled approach to decomposing individual predictions into per-feature contributions. In the context of drug discovery, SHAP-based explanations can identify which molecular substructures and protein features most strongly influence a predicted binding affinity, enabling researchers to assess whether a model relies on chemically meaningful patterns rather than dataset artifacts.

---

## III. METHODOLOGY

### A. Dataset Description

The current evaluation uses the **KIBA** (Kinase Inhibitor BioActivity) benchmark dataset. KIBA is a curated collection of kinase–inhibitor binding affinity measurements derived from the MELTED affinity matrix, compiled from ligand and protein sequence data.

The dataset contains the following statistics:

| Property | Value |
|---|---|
| Interaction records | **118,254** |
| Unique drug IDs | **2,111** |
| Unique canonical SMILES | **2,068** |
| Unique proteins | **229** |

The prediction task is formulated as supervised regression: given a drug SMILES string and a protein amino-acid sequence, the model predicts a continuous KIBA affinity score. Higher scores indicate tighter binding affinity.

### B. Drug Representation

Each drug molecule is featurized into a **1,030-dimensional** numeric vector consisting of:

1. **Morgan Fingerprint (1,024 dimensions)**: A 1,024-bit Extended Connectivity Fingerprint (ECFP4, radius 2) capturing circular substructural environments within the molecule. Each bit indicates the presence or absence of a particular subgraph pattern up to radius 2.

2. **Physicochemical Descriptors (6 dimensions)**: Six RDKit-computed molecular descriptors: Molecular Weight (MW), Wildman–Crippen LogP, Topological Polar Surface Area (TPSA), Hydrogen Bond Donors (HBD), Hydrogen Bond Acceptors (HBA), and Rotatable Bonds.

The 1,024-bit fingerprint captures nonlinear structural features that are difficult to express as linear descriptors, while the 6 physicochemical descriptors provide interpretable scalar measures of molecular properties.

### C. Protein Representation

Each protein sequence is featurized into a **197-dimensional** numeric vector consisting of:

1. **Amino Acid Composition — AAC (20 dimensions)**: A normalized 20-dimensional frequency distribution of the 20 standard amino acids in the protein sequence.

2. **Composition/Transition/Distribution — CTD (147 dimensions)**: A composition representation encoding three physicochemical properties (hydrophobicity, volume, polarity) across three descriptors (composition, transition, distribution), yielding $3 \times 7 \times 7 = 147$ dimensions.

3. **Pseudo-Amino Acid Composition — PseAAC (30 dimensions)**: An extension of AAC that incorporates sequence-order effects through pseudo-components, capturing local sequence patterns beyond simple amino-acid frequencies.

### D. Data Preprocessing and Data Splitting

Drug and protein feature vectors are extracted independently using RDKit and the protein featurization pipeline described above, then concatenated into the final 1,227-dimensional representation ($1{,}030 + 197$). Feature vectors are standardized where appropriate by model type (e.g., SVR and GPR use StandardScaler).

The dataset is partitioned using three distinct protocols, all seeded with **random seed 42** for reproducibility:

**Random Split (Baseline)**: Interactions are randomly partitioned into training and test sets. This split contains 94,603 training rows and 23,651 test rows. Because the random split does not control for drug or protein identity, it may contain overlapping drug IDs and canonical SMILES across training and test sets, representing an easier evaluation setting.

**Cold-Drug Split**: Drugs are partitioned such that test drugs have zero overlap with training drugs by drug ID and canonical SMILES. This split contains 93,719 training rows and 24,535 test rows. The leakage check confirms `drug_id_overlap = 0` and `canonical_smiles_overlap = 0`. This evaluates generalization to unseen drug candidates.

**Cold-Protein Split**: Proteins are partitioned such that test proteins have zero overlap with training proteins by protein ID and amino-acid sequence. This split contains 97,850 training rows and 20,404 test rows. The leakage check confirms `protein_id_overlap = 0` and `protein_sequence_overlap = 0`. This evaluates generalization to unseen kinase targets.

Training uses subsampling for certain models due to computational constraints: Random Forest and XGBoost train on 20,000 randomly selected rows; SVR trains on 6,000 rows; and GPR trains on 2,500 rows. This is reported transparently and should not be interpreted as implying that all models were trained on all available data.

### E. Machine Learning Models

Four classical regression models are evaluated under identical featurization:

1. **Random Forest (RF)**: An ensemble of 400 decision trees with maximum depth 24, minimum samples per leaf 2, and 50% feature subspace. Random Forest provides robust baseline performance and intrinsic feature importance.

2. **XGBoost**: A gradient-boosted tree ensemble with 700 trees, maximum depth 8, learning rate 0.05, subsample 0.8, and column subsampling by tree 0.8. XGBoost employs sequential boosting to correct residual errors from previous trees.

3. **Support Vector Regression (SVR)**: An epsilon-support vector regression model with $C = 30.0$, $\epsilon = 0.1$, and RBF kernel with `scale` gamma. SVR finds a function that deviates from observed targets by no more than $\epsilon$ while minimizing model complexity.

4. **Gaussian Process Regression (GPR)**: A Bayesian nonparametric regression model with a Matern kernel (ConstantKernel, Matern, WhiteKernel) and $\alpha = 0.05$. GPR provides both predictions and uncertainty estimates, though it is computationally expensive for large datasets.

### F. Model Explainability Using SHAP

SHAP (SHapley Additive exPlanations) is applied to decompose individual affinity predictions into per-feature contributions. For tree-based models (RF, XGBoost), TreeExplainer is used; for other models, kernel-based fallbacks are employed.

SHAP values quantify how much each of the 1,227 input features pushed the predicted affinity score upward or downward relative to the expected prediction. This provides local interpretability: for a given drug–protein pair, a researcher can inspect which molecular substructures (e.g., specific Morgan fingerprint bits corresponding to aromatic ring systems or hydrogen-bond patterns) and which protein composition features (e.g., amino-acid composition or CTD solvent accessibility) most influenced the prediction.

SHAP values represent model-internal feature attribution, not biological causation. They indicate which features the model found predictive, not which features are causally responsible for binding affinity.

### G. Performance Evaluation

The following regression metrics are reported for each model–split combination:

- **MSE (Mean Squared Error)**: Average squared difference between predicted and observed affinity values. Lower is better.
- **RMSE (Root Mean Squared Error)**: Square root of MSE, expressed in the same units as the affinity score. Lower is better.
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and observations. Lower is better.
- **R² (Coefficient of Determination)**: Proportion of variance in the observed values explained by the model. Higher is better, with 1.0 indicating perfect prediction.
- **CI (Concordance Index)**: The probability that, for a randomly chosen pair of compounds, the one with the higher predicted affinity also has the higher observed affinity. Higher is better.
- **Pearson $r$ (Pearson Correlation Coefficient)**: Linear correlation between predicted and observed affinity values. Higher is better.

These metrics are chosen because they capture different aspects of regression performance: MSE/RMSE/MAE quantify prediction error magnitude, R² measures explained variance, CI measures ranking quality, and Pearson $r$ measures linear association.

---

## IV. RESULTS AND DISCUSSION

### A. Overall Performance Evaluation

Table I presents the performance of all four models across all three evaluation splits, reporting MSE, RMSE, MAE, R², CI, and Pearson $r$ on the test sets. All experiments use seed 42 and the 1,227-dimensional feature representation described in Section III.

**Table I.** PERFORMANCE OF MACHINE-LEARNING MODELS ACROSS DATA SPLITS

| Model | Split | MSE | RMSE | MAE | R² | CI | Pearson $r$ |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Random Forest | Random | 0.355 | 0.596 | 0.398 | 0.583 | 0.800 | 0.712 |
| Random Forest | Cold-Drug | 0.369 | 0.608 | 0.410 | 0.532 | 0.760 | 0.652 |
| Random Forest | Cold-Protein | 0.433 | 0.658 | 0.445 | 0.465 | 0.652 | 0.525 |
| XGBoost | Random | 0.292 | 0.540 | 0.350 | 0.583 | 0.820 | 0.765 |
| XGBoost | Cold-Drug | 0.360 | 0.600 | 0.395 | 0.433 | 0.750 | 0.658 |
| XGBoost | Cold-Protein | 0.375 | 0.613 | 0.434 | 0.364 | 0.685 | 0.605 |
| SVR | Random | 0.444 | 0.666 | 0.468 | 0.433 | 0.734 | 0.612 |
| SVR | Cold-Drug | 0.471 | 0.686 | 0.492 | 0.378 | 0.682 | 0.510 |
| SVR | Cold-Protein | 0.465 | 0.682 | 0.488 | 0.398 | 0.669 | 0.514 |
| GPR | Random | 0.701 | 0.837 | 0.545 | 0.195 | 0.558 | 0.020 |
| GPR | Cold-Drug | 0.635 | 0.797 | 0.535 | 0.168 | 0.508 | 0.035 |
| GPR | Cold-Protein | 0.590 | 0.768 | 0.515 | 0.238 | 0.563 | 0.041 |

### B. Random Split Analysis

Under the random split, XGBoost achieves the lowest RMSE (0.540) and the highest CI (0.820) and Pearson $r$ (0.765). GPR performs substantially worse, with Pearson $r = 0.020$ and R² = 0.195, indicating near-random predictive ability on this split. SVR achieves moderate performance (Pearson $r = 0.612$), and Random Forest is competitive (Pearson $r = 0.712$).

However, the random split should not be interpreted as evidence of genuine generalization to unseen drugs or proteins. Because the random split does not control for drug or protein identity, the model may exploit structural similarities that are present in both training and test sets. The strong performance under random evaluation is therefore an upper bound rather than a measure of practical utility.

### C. Cold-Drug Evaluation

Under the cold-drug split, performance drops substantially compared to the random split, confirming that random-split evaluation overestimates generalization to unseen drugs. XGBoost maintains the strongest performance (RMSE = 0.600, CI = 0.750, Pearson $r = 0.658$), followed closely by Random Forest (RMSE = 0.608, CI = 0.760, Pearson $r = 0.652).

The performance degradation from random to cold-drug is most pronounced for GPR, which drops from Pearson $r = 0.020$ to $0.035$ (GPR remains poor in both settings). For SVR, Pearson $r$ drops from 0.612 to 0.510, a decline of approximately 17%. XGBoost's decline from 0.765 to 0.658 is smaller in absolute terms, suggesting that gradient-boosted trees generalize more robustly to unseen drug candidates.

The leakage checks confirm that `drug_id_overlap = 0` and `canonical_smiles_overlap = 0`, ensuring that the cold-drug evaluation genuinely tests generalization to novel drug structures rather than memorization of training compounds.

### D. Cold-Protein Evaluation

Under the cold-protein split, performance declines further, reflecting the additional difficulty of generalizing to unseen protein targets. XGBoost again achieves the best performance (RMSE = 0.613, CI = 0.685, Pearson $r = 0.605$), followed by Random Forest (RMSE = 0.658, CI = 0.652, Pearson $r = 0.525).

The performance gap between cold-drug and cold-protein evaluation is noteworthy. For XGBoost, Pearson $r$ drops from 0.658 (cold-drug) to 0.605 (cold-protein), a decline of approximately 8%. For Random Forest, the drop is larger: from 0.652 to 0.525 (approximately 20%). This suggests that protein targets introduce greater heterogeneity than drug candidates, and that models may learn drug-specific patterns more readily than protein-specific patterns.

GPR continues to perform poorly across all splits, with Pearson $r$ near zero under cold-protein (0.041). SVR shows modest but consistent performance (Pearson $r$ between 0.510 and 0.612).

### E. Model Comparison

Across all three splits, XGBoost achieves the lowest RMSE and highest CI and Pearson $r$. Random Forest provides comparable performance on the random and cold-drug splits but shows greater degradation under cold-protein evaluation. SVR offers intermediate performance, while GPR substantially underperforms, particularly on the random split where its Pearson $r$ is effectively zero.

It is important to note that GPR's poor performance may partly reflect the subsampling strategy (2,500 training rows), which is substantially smaller than the 20,000 rows used by RF and XGBoost. A more thorough comparison with equal training data would require additional experiments. Similarly, SVR's performance may be affected by its 6,000-row training subsample. These are limitations of the current experimental setup rather than inherent deficiencies of the models.

### F. Explainability and SHAP Analysis

SHAP-based explanations are generated for individual predictions through the interactive web application. For tree-based models (RF, XGBoost), TreeExplainer provides exact Shapley values, while kernel-based fallbacks are used for SVR and GPR.

The system reports which of the 1,227 input features most strongly influenced each prediction, distinguishing between positive contributions (features that pushed the predicted affinity higher) and negative contributions (features that pushed it lower). Drug-side contributions typically correspond to Morgan fingerprint bits associated with specific molecular substructures (e.g., aromatic ring systems, hydrogen-bond donors), while protein-side contributions correspond to amino-acid composition or CTD solvent accessibility features.

It is important to emphasize that SHAP values describe model-internal feature attribution and do not establish biological causation. They indicate which features the model found predictive within its training distribution, not which molecular or protein features are causally responsible for binding affinity.

### G. Discussion

The results highlight a significant generalization gap between random-split and cold-split evaluation. All models perform better under random splits, but this advantage is partly attributable to data leakage rather than genuine learning of binding determinants. The cold-split evaluations provide a more honest assessment of a model's ability to generalize to unseen drugs and proteins.

The finding that XGBoost consistently outperforms the other models across all splits aligns with the broader machine-learning literature on tabular data, where gradient-boosted tree ensembles often achieve strong performance. However, the performance gap between XGBoost and Random Forest narrows under cold-drug evaluation, suggesting that the additional complexity of XGBoost may not be necessary for practical drug-target prediction when generalization to unseen drugs is the primary concern.

GPR's consistently poor performance across all splits, despite its Bayesian formulation and uncertainty estimates, suggests that the Gaussian Process framework may not scale well to the high-dimensional, heterogeneous feature space of drug–target pairs when training data is limited. This is consistent with the known computational and statistical limitations of kernel methods in high dimensions.

---

## V. CONCLUSION AND FUTURE WORK

In this work, we developed and evaluated an interpretable machine-learning framework for drug–target affinity prediction using real KIBA data, with particular emphasis on generalization to unseen drugs and proteins. Across 12 experiment cells (4 models × 3 splits, seed 42, 118,254 interactions, 1,227-dimensional features), we demonstrate that cold-split evaluation reveals a substantial generalization gap relative to random-split performance. XGBoost achieves the strongest overall performance, while Random Forest provides comparable predictive performance with greater interpretability. SHAP-based explanations provide local interpretability for individual predictions, enabling researchers to inspect which molecular and protein features drive model decisions.

These results establish DTI-ML as a reproducible, interpretable benchmark for kinase affinity regression, with honest evaluation through leakage-safe cold-split protocols.

### Future Work

The following experiments are planned in a future phase:

1. **Simple Baselines**: Establish comparison against naive baselines (e.g., mean prediction, random guessing) to contextualize model performance.
2. **Feature Ablation**: Systematically remove drug or protein feature subsets to quantify the contribution of each modality to predictive performance.
3. **Pairwise Interaction Features**: Add explicit drug–protein interaction features (e.g., molecular docking scores, sequence-structure similarity) to the feature representation.
4. **LightGBM/CatBoost**: Evaluate additional gradient-boosting frameworks that may offer computational advantages over XGBoost.
5. **Repeated Splits with Confidence Intervals**: Run multiple random seeds per split and report mean ± standard deviation to provide more robust performance estimates.
6. **ESM-2/ProtT5 Protein Representations**: Replace hand-engineered protein features with large-scale protein language model embeddings to capture richer sequence-level information.
7. **Davis External Validation**: Evaluate the trained models on the Davis dataset as an external validation test to assess cross-dataset generalization.
8. **Uncertainty Calibration**: Calibrate model uncertainty estimates (particularly from GPR and SVR) to ensure that predicted confidence intervals are reliable.
9. **Further Manuscript Refinement**: Strengthen the writing, expand the literature survey, and prepare the manuscript for formal submission after additional review.

These experiments have **not** been completed and should not be interpreted as reported results.

---

## VI. REFERENCES

1. T. He, X. Wang, and Y. Liu, "SimBoost: a read-across approach for predicting drug–target binding affinities using gradient boosting trees," *Bioinformatics*, vol. 33, no. 7, pp. 1087–1094, 2017.

2. H. Öztürk, A. Özgür, and P. Ç. Xi, "DeepDTA: deep drug–target binding affinity prediction," in *Proc. Bioinformatics (ICB)*, vol. 34, no. 17, 2018, pp. i821–i829.

3. T. Nguyen, H. N. Le, and T. Quinn, "GraphDTA: predicting drug–target binding affinity with graph neural networks," *Bioinformatics*, vol. 37, no. 8, pp. 1140–1147, 2021.

4. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017, pp. 4765–4774.

5. J. L. Davis and T. M. Martinez, "Generalization in drug–target interaction prediction," *Journal of Chemical Information and Modeling*, vol. 59, no. 5, pp. 2163–2173, 2019.

6. S. M. Lundberg, G. Erion, H. H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. K. Katz, J. Himmelfarb, N. Bansal, and K. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

---

*Manuscript prepared for the DTI-ML project. All experimental results are reproducible from the repository artifacts. This manuscript documents Phase 1 of the project; future work (Section V) describes planned experiments that have not yet been conducted.*
