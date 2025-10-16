from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from model_service import ModelService

app = FastAPI(title="Assignment3 Backend", version="0.1.0")

# Allow the React dev server
origins = [
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

class PredictRequest(BaseModel):
  code: str

class PredictResponse(BaseModel):
  prediction: Any
  model_name: str

class SelectModelRequest(BaseModel):
  model_name: str

model_service = ModelService()

@app.get("/health")
def health():
  return {"status": "ok"}

@app.get("/models")
def list_models():
  return {"available_models": model_service.available_models(), "current_model": model_service.current_model_name}

@app.put("/model")
def select_model(req: SelectModelRequest):
  try:
    model_service.select_model(req.model_name)
    return {"status": "ok", "model_name": model_service.current_model_name}
  except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
  try:
    pred = model_service.predict(req.code)
    return PredictResponse(prediction=pred, model_name=model_service.current_model_name)
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
