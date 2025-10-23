from typing import Any
from pydantic import BaseModel

class PredictRequest(BaseModel):
  code: str

class PredictResponse(BaseModel):
  model_config = {"protected_namespaces": ()}
  prediction: Any
  model_name: str

class SelectModelRequest(BaseModel):
  model_config = {"protected_namespaces": ()}
  model_name: str