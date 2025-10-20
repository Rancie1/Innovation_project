from typing import Any
from pydantic import BaseModel



class PredictRequest(BaseModel):
  code: str

class PredictResponse(BaseModel):
  prediction: Any
  model_name: str

class SelectModelRequest(BaseModel):
  model_name: str