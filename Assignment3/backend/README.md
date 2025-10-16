# Assignment3 FastAPI Backend

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Docs: http://localhost:8000/docs
- Health: GET http://localhost:8000/health
- List models: GET http://localhost:8000/models
- Select model: PUT http://localhost:8000/model {"model_name": "baseline"}
- Predict: POST http://localhost:8000/predict {"code": "print('hello')"}

## Notes
- CORS allows the React dev server at http://localhost:3000.
- `model_service.py` is a stub; replace with Assignment2 model inference.
