# Assignment 3 — Full‑Stack AI App (React + FastAPI)

This assignment delivers a full‑stack web app where users submit code snippets and receive AI model predictions. The frontend is built with React, and the backend is implemented with FastAPI. Models are trained or loaded from Assignment 2 datasets.

## Repo structure

```
Assignment3/
  backend/          # FastAPI app, model training/loading, saved models
  frontend/         # React app (form, validation, API integration)
```

## Backend (FastAPI)

### 1) Setup
```bash
cd /Users/nathanrancie/Desktop/Innovation/Innovation_project/Assignment3/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train models (optional if artifacts already exist)
- Assignment2 Model‑2 (multiclass CWE categories):
```bash
python model_loader.py
```
- Assignment2 Model‑1 (binary, alternative dataset):
```bash
python model1_loader.py
```
Model artifacts are saved in `backend/models/` (e.g., `logistic_regression.pkl`, `random_forest.pkl`, `model1_logreg.pkl`, `model1_random_forest.pkl`, plus metadata like `label_encoder.pkl`, `target_names.pkl`, `model1_classes.pkl`).

### 3) Run the API
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### API endpoints
- GET `/health` → `{ "status": "ok" }`
- GET `/models` → `{ available_models, current_model }`
- PUT `/model` body: `{ "model_name": "logistic_regression" }` → switch active model
- POST `/predict` body: `{ "code": "print('hello')" }` → returns prediction JSON (category, confidence, probabilities, etc.)

Example usage:
```bash
# List models
curl http://localhost:8000/models

# Select a model
curl -X PUT http://localhost:8000/model \
  -H "Content-Type: application/json" \
  -d '{"model_name":"logistic_regression"}'

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"code":"print(\"hello\")"}'
```

### Notes
- CORS is enabled for `http://localhost:3000` so the React dev server can call the API.
- If you change model artifacts, restart the API to pick them up.

## Frontend (React)

### 1) Setup and run
```bash
cd /Users/nathanrancie/Desktop/Innovation/Innovation_project/Assignment3/frontend
npm install
npm start
```
- App will run at http://localhost:3000
- If your backend runs on a different URL, create `.env` in `frontend/`:
```
REACT_APP_API_URL=http://localhost:8000
```
Then restart `npm start`.

### Features
- Responsive, accessible UI
- Code snippet textarea with validation (min length)
- Model dropdown populated via `/models`
- Calls `POST /predict` and renders the JSON result
- Error and loading states

## Architecture overview

- `backend/main.py` — FastAPI app (routes, CORS, wiring)
- `backend/models/models.py` — Pydantic request/response models
- `backend/model_service.py` — Model selection, prediction, response shaping
- `backend/model_loader.py` — Trains/loads Assignment2 Model‑2 (multiclass)
- `backend/model1_loader.py` — Trains/loads Assignment2 Model‑1 (binary)
- `backend/models/*.pkl` — Saved estimators and metadata
- `frontend/src/App.js` — UI (form, model select, results)
- `frontend/src/api.js` — Minimal API client

## Troubleshooting

- 500 on /predict; Pydantic serialization error (NumPy types):
  - Fixed by converting NumPy scalars to native Python in `model_service.py`.

- Warning: InconsistentVersionWarning (scikit‑learn pickles):
  - Ensure your venv has the same scikit‑learn version as used to train models.
  - Current pin: `scikit-learn==1.6.1` in `backend/requirements.txt`.
  - Reinstall or retrain as needed, then restart the API.

- `model1_classes` shows as a selectable model:
  - Fixed by excluding non‑estimator artifacts from loader.
  - If still present, fully stop and restart the server.

- `FileNotFoundError: data_model1/...` when training model‑1:
  - `model1_loader.py` now uses paths relative to its file; run from `backend/` and ensure `backend/data_model1/` exists with `train.csv`/`test.csv` or the `.parquet` equivalents.

## Grading rubric alignment
- At least 2 HTTP methods used (GET, PUT, POST) ✔️
- Integrates AI models from Assignment 2 via FastAPI ✔️
- Frontend form with validation and responsive UI ✔️
- Interactive visualization-friendly results (JSON displayed; extendable) ✔️

## License / Authors
- Built by Nathan Rancie for Assignment 3.
