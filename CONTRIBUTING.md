# Contributing to DTI-ML

## Source of Truth

- [PRODUCT_DESIGN.md](PRODUCT_DESIGN.md) defines the production UI and interaction contract.
- [design-reference.html](design-reference.html) is the visual reference.
- [README.md](README.md) defines setup, commands, and deployment.
- `docs/archive/` contains historical planning material and is not an active task tracker.

Do not change the product design, page order, visual tokens, or API behavior casually. Record intentional deviations in `PRODUCT_DESIGN.md`.

## Module Ownership

### Backend

- `backend/main.py`: FastAPI app, middleware, and thin route handlers.
- `backend/config.py`: environment-derived settings.
- `backend/schemas.py`: Pydantic request and response contracts.
- `backend/prediction_service.py`: model loading, validation, inference, and response shaping.
- `backend/chat_service.py`: chatbot guardrails, rate limiting, cache, and provider calls.
- `backend/tests/`: API and service tests.
- `features/`: drug and protein feature extraction.
- `evaluation/`: metrics and SHAP utilities.
- `app/helpers.py`: reusable model, preset, and benchmark helpers.

Route handlers should not contain feature extraction or model logic.

### Frontend

- `frontend/src/main.jsx`: bootstrap only.
- `frontend/src/app/App.jsx`: page state and application composition.
- `frontend/src/app/api.js`: all HTTP requests and API errors.
- `frontend/src/app/constants.js`: routes, models, and fallback constants.
- `frontend/src/components/`: reusable visual and interactive pieces.
- `frontend/src/pages/`: page composition only.
- `frontend/src/content/copy.js`: static product copy and content data.
- `frontend/src/styles.css`: shared design tokens and responsive styles.

Pages compose components. Components should not fetch data directly. Do not add a state-management library for local page state.

## Common Changes

### Add a page

1. Add a page component under `frontend/src/pages/`.
2. Add its label to `PAGES` in `frontend/src/app/constants.js`.
3. Register it in `frontend/src/app/App.jsx`.
4. Add the page requirements to `PRODUCT_DESIGN.md` before implementation.
5. Add navigation coverage to the frontend tests.

### Add a component

1. Create it under `frontend/src/components/`.
2. Keep animation and interaction state inside the component that owns it.
3. Reuse existing CSS tokens and classes.
4. Add focused tests for non-trivial behavior.

### Add an API endpoint

1. Add request/response models in `backend/schemas.py`.
2. Add service logic in the owning backend service module.
3. Keep the route in `backend/main.py` thin.
4. Add a `backend/tests/` test for success and failure behavior.
5. Update the API contract in `PRODUCT_DESIGN.md` or `README.md`.

### Add a model or preset

- Add checkpoint mapping and validation in `backend/prediction_service.py`.
- Keep preset metadata separate for drug name, target name, SMILES, and sequence.
- Add or update API tests.
- Do not hardcode model behavior in React.

## Configuration

- Copy `.env.example` to `.env` for backend settings.
- Copy `frontend/.env.example` to `frontend/.env` for `VITE_API_BASE`.
- Never commit `.env`, API keys, model secrets, generated data, `node_modules`, or build output.
- Update CORS origins through `DTI_ALLOWED_ORIGINS` rather than hardcoding deployed domains.

## Validation

Run before finishing a change:

```powershell
.\venv\Scripts\python.exe -m compileall backend app/helpers.py features evaluation
.\venv\Scripts\python.exe -m pytest -q

cd frontend
npm ci
npm test -- --run
npm run build
npm run test:e2e

cd ..
git diff --check
git status --short
```

Browser screenshots are required before claiming visual parity with the design reference. Check desktop and mobile viewports listed in `PRODUCT_DESIGN.md`. If browser tooling is unavailable, report that limitation.

## Change Discipline

- Do not restore Streamlit.
- Do not retrain models for UI-only changes.
- Do not move scientific modules without updating imports and tests.
- Do not use destructive Git commands.
- Do not create commits unless explicitly requested.
- Keep changes focused and update documentation when contracts or startup commands change.
