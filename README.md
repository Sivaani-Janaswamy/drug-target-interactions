# DTI-ML

DTI-ML predicts kinase drug-target binding affinity from a drug SMILES string and a protein sequence. It uses classical machine learning, leakage-safe cold-split evaluation, SHAP-based explainability, and a project-grounded AI chatbot.

The production application is a React frontend backed by a FastAPI service. The required product design is defined by [PRODUCT_DESIGN.md](PRODUCT_DESIGN.md) and [design-reference.html](design-reference.html).

## Research Foundation

### Dataset

- **Real KIBA** (Kinase Inhibitor BioActivity benchmark dataset)
- **118,254** drug-target interactions
- **2,111** unique drugs, **2,068** unique canonical SMILES
- **229** unique proteins

### Feature Representation

- **1,030** drug features: Morgan ECFP4 fingerprints (1,024-bit) + 6 physicochemical descriptors (MW, LogP, TPSA, HBD, HBA, Rotatable Bonds)
- **197** protein features: AAC (20-dim) + CTD (147-dim) + PSEAAC (30-dim)
- **1,227** combined dimensions

### Evaluation

Three partitioning protocols evaluated across four classical ML models:

| Split | Purpose | Leakage Prevention |
|---|---|---|
| **Random Split (Baseline)** | Standard evaluation | N/A |
| **Cold-Drug Split** | Unseen drug candidates | `drug_id_overlap = 0`, `canonical_smiles_overlap = 0` |
| **Cold-Protein Split** | Unseen protein targets | `protein_id_overlap = 0`, `protein_sequence_overlap = 0` |

- **Models**: Random Forest, XGBoost, Support Vector Regression (SVR), Gaussian Process Regression (GPR)
- **12** completed experiment cells (3 splits × 4 models)
- **seed = 42** for reproducibility
- **Metrics**: MSE, RMSE, Pearson _r_, Concordance Index (CI)

Authoritative current results are in `models/kiba_results.csv`, `models/kiba_detailed_results.csv`, and `models/kiba_experiment_manifest.json`.

### Legacy Synthetic Data

Legacy synthetic/development benchmark data may exist in the repository for testing and development purposes. It is **NOT** the current research evaluation. The current evaluation uses real KIBA data with the 1,227-dimensional representation and leakage-safe cold splits described above.

## Architecture

```mermaid
%% DTI-ML Architecture
graph TD
    User((User))

    subgraph React Frontend
        PredictionUI[Prediction UI]
        ChatWidget[Chat Widget]
        SciencePage[Model & Science Page]
    end

    subgraph FastAPI Backend
        API[POST /api/chat]
        Predict[POST /api/predict]
        Benchmarks[GET /api/benchmarks]
    end

    subgraph Knowledge & Data Layer
        Manifest[kiba_experiment_manifest.json]
        Summary[data/kiba_summary.json]
        SplitMeta[data/kiba_split_metadata.json]
        Features[features/feature_manifest.json]
    end

    subgraph ML Models
        RF[Random Forest]
        XGB[XGBoost]
        SVR[SVR]
        GPR[GPR]
    end

    User --> PredictionUI
    User --> ChatWidget
    PredictionUI --> Predict
    ChatWidget --> API
    SciencePage --> Benchmarks
    Predict --> Features
    Predict --> RF & XGB & SVR & GPR
    API --> Knowledge & Data Layer
    Knowledge & Data Layer -->|"Gemini 2.5 Flash"| API
    Benchmarks --> Manifest
    RF --> Predict
    XGB --> Predict
    SVR --> Predict
    GPR --> Predict
```

### Data and Model Layer

- **KIBA data**: Real kinase inhibitor bioactivity data
- **Feature representations**: Drug (1,030-dim) and protein (197-dim) feature vectors
- **Trained model artifacts**: Checkpoints under `models/artifacts/` for each model and split
- **Experiment results/manifest**: `models/kiba_experiment_manifest.json` with full results matrix and best models per split

### Chatbot

The chatbot provides project-grounded Q&A:

- **POST** `/api/chat` with a question and optional prediction context
- **Project-knowledge retrieval** from `kiba_experiment_manifest.json`, `kiba_summary.json`, `kiba_split_metadata.json`, and `feature_manifest.json`
- **Gemini 2.5 Flash** generates grounded answers with source references
- **Rate limiting**: 18 messages per session per hour
- **Guardrails**: Blocked-pattern filter, refusal for off-topic/medical questions, fallback when no API key
- **Local cache** for common questions (score meaning, cold-split rationale, SHAP explanation)

The chatbot uses lightweight keyword-based knowledge retrieval — not vector databases, embeddings, or LangChain.

## Repository Guide

```text
backend/       FastAPI routes, services, schemas, and tests
frontend/      React production interface
features/      Drug and protein feature extraction
evaluation/    Metrics and interpretability
models/        Checkpoints, benchmark results, and experiment manifest
data/          Dataset summaries, split metadata, and source data
paper/         IEEE manuscript and figures
docs/archive/  Historical planning documents
```

Use [CONTRIBUTING.md](CONTRIBUTING.md) for extension and maintenance rules. Use [PRODUCT_DESIGN.md](PRODUCT_DESIGN.md) for UI and interaction requirements.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm ci
```

Copy `.env.example` to `.env` and `frontend/.env.example` to `frontend/.env`. Configure `VITE_API_BASE` for the frontend and `DTI_ALLOWED_ORIGINS`, `DTI_MODELS_DIR`, `DTI_API_VERSION`, and optional `GEMINI_API_KEY` for the API.

## Run Locally

Start the API from the repository root:

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```powershell
cd frontend
npm run dev
```

The default URLs are `http://127.0.0.1:8000` for the API and `http://localhost:5173` for the frontend.

## Tests and Build

```powershell
# Repository root
.\venv\Scripts\python.exe -m compileall backend app/helpers.py features evaluation
.\venv\Scripts\python.exe -m pytest -q

# frontend/
npm ci
npm test -- --run
npm run build
npm run test:e2e
```

Playwright E2E tests start the local FastAPI and Vite servers automatically. The successful prediction flow uses the real local model; only deterministic error cases use API interception. Install Chromium once with `npx playwright install chromium` when browser tooling is available. E2E screenshots and reports are written to ignored test-output directories.

## Data and Models

Run the data pipeline and training scripts only when intentionally regenerating artifacts:

```powershell
python data/build_kiba_dataset.py
python features/featurizer.py
python models/train_models.py
```

Model checkpoints are ignored by Git and must be present under `DTI_MODELS_DIR` at runtime. Authoritative current result files are `models/kiba_results.csv`, `models/kiba_detailed_results.csv`, and `models/kiba_experiment_manifest.json`.

## Production

Build the frontend and run the API with production hosts:

```powershell
cd frontend
npm ci
npm run build

cd ..
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Serve `frontend/dist` from a static host or reverse proxy. Add the deployed frontend origin to `DTI_ALLOWED_ORIGINS`. Deployment is currently manual; no hosted environment or CI/CD pipeline is included.

`frontend/package-lock.json` and `requirements-lock.txt` provide reproducible dependency snapshots. Regenerate the Python snapshot only after intentional dependency changes:

```powershell
python -m pip install -r requirements.txt
python -m pip freeze > requirements-lock.txt
```

## Explainability

SHAP-based feature attribution is available for individual predictions. The system reports which drug substructure features (Morgan fingerprint bits, physicochemical descriptors) and protein composition features (AAC, CTD, PSEAAC) contributed most to the predicted binding affinity, with both positive and negative contributions identified.
