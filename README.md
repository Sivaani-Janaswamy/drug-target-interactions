# DTI-ML

Drug-target interaction affinity prediction using classical machine learning with cold-split evaluation and interpretability.

## Project structure

- `data/` — datasets, raw files, processed splits
- `features/` — extracted drug and protein features
- `models/` — trained models and checkpoints
- `app/` — web application entrypoints
- `paper/` — manuscript drafts and figures

## Setup

1. Create a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` if needed.

## Workflow

- Data collection and cleaning in `data/`
- Feature generation in `features/`
- Model training in `models/`
- App work in `app/`
- Paper development in `paper/`
