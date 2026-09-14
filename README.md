# DTI-ML

DTI-ML predicts kinase drug-target binding affinity from a drug SMILES string and a protein sequence. It uses classical machine learning, cold-split evaluation, and SHAP-based explanations.

The production application is a React frontend backed by a FastAPI service. The required product design is defined by [PRODUCT_DESIGN.md](PRODUCT_DESIGN.md) and [design-reference.html](design-reference.html).

## Architecture

```text
React frontend
    |
    v
FastAPI API
    |
    +-- backend/config.py       environment settings
    +-- backend/schemas.py      request/response contracts
    +-- backend/prediction_service.py
    +-- backend/chat_service.py
    |
    +-- features/               RDKit and protein features
    +-- evaluation/             metrics and SHAP utilities
    +-- models/                 checkpoints and benchmark results
```

## Repository Guide

```text
backend/       FastAPI routes, services, schemas, and tests
frontend/      React production interface
features/      Drug and protein feature extraction
evaluation/    Metrics and interpretability
models/        Training scripts, checkpoints, and results
data/          Dataset builders, splits, and source data
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
```

## Data and Models

Run the data pipeline and training scripts only when intentionally regenerating artifacts:

```powershell
python data/build_kiba_dataset.py
python features/featurizer.py
python models/train_models.py
```

Model checkpoints are ignored by Git and must be present under `DTI_MODELS_DIR` at runtime. The checked-in `models/results.csv` supplies the science-page benchmark table.

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
