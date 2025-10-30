import os
import pickle
from typing import List, Dict, Any
from model_loader import ModelLoader

class ModelService:
  def __init__(self) -> None:
    self._available: List[str] = ["baseline", "logistic_regression", "random_forest", "model1_logreg", "model1_random_forest"]
    self.current_model_name: str = "baseline"
    self.current_model = None
    self.label_encoder = None
    self.target_names = []
    self.model1_classes = None
    self.model_loader = ModelLoader()
    
    # Try to load pre-trained models
    self._load_models()

  def _load_models(self):
    """Load pre-trained models if they exist."""
    try:
      models = self.model_loader.load_trained_models()
      dynamic = []
      if models:
        dynamic += list(models.keys())
        self.label_encoder = self.model_loader.label_encoder
        self.target_names = self.model_loader.target_names
      # Model-1 artifacts (optional)
      if os.path.exists("models/model1_logreg.pkl"):
        dynamic.append("model1_logreg")
      if os.path.exists("models/model1_random_forest.pkl"):
        dynamic.append("model1_random_forest")
      if os.path.exists("models/model1_classes.pkl"):
        with open("models/model1_classes.pkl", "rb") as f:
          self.model1_classes = pickle.load(f)
      if dynamic:
        self._available = ["baseline"] + sorted(set(dynamic), key=str)
        print(f"Loaded pre-trained models: {self._available}")
      else:
        print("No pre-trained models found. Using baseline model.")
    except Exception as e:
      print(f"Could not load pre-trained models: {e}. Using baseline model.")

  def available_models(self) -> List[str]:
    return list(self._available)

  def select_model(self, model_name: str) -> None:
    if model_name not in self._available:
      raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(self._available)}")
    
    self.current_model_name = model_name
    
    # Load the selected model
    if model_name != "baseline":
      try:
        with open(f"models/{model_name}.pkl", "rb") as f:
          self.current_model = pickle.load(f)
        print(f"Loaded model: {model_name}")
      except Exception as e:
        print(f"Could not load model {model_name}: {e}")
        self.current_model = None
    else:
      self.current_model = None

  def _py_int(self, value: Any) -> Any:
    try:
      return int(value)
    except Exception:
      return value

  def _format_prediction(self, prediction: Any, proba: List[float] | None) -> Dict[str, Any]:
    # Model-2 style labels
    if self.current_model_name in ("logistic_regression", "random_forest") and self.target_names:
      predicted_category = self.target_names[self._py_int(prediction)] if isinstance(prediction, (int,)) or str(prediction).isdigit() else self.target_names[int(prediction)]
      return {
        "predicted_category": predicted_category,
        "confidence": float(max(proba)) if proba is not None else None,
        "all_probabilities": {name: float(p) for name, p in zip(self.target_names, (proba or []))},
        "model_used": self.current_model_name,
      }
    # Model-1 binary labels
    if self.current_model_name in ("model1_logreg", "model1_random_forest") and self.model1_classes is not None:
      py_label = self._py_int(prediction)
      return {
        "predicted_label": py_label if isinstance(py_label, int) else str(py_label),
        "predicted_label_name": str(prediction),
        "classes": list(map(str, self.model1_classes)),
        "confidence": float(max(proba)) if proba is not None else None,
        "model_used": self.current_model_name,
      }
    # Fallback generic
    py_pred = self._py_int(prediction)
    return {
      "predicted": py_pred if isinstance(py_pred, int) else str(py_pred),
      "confidence": float(max(proba)) if proba is not None else None,
      "model_used": self.current_model_name,
    }

  def predict(self, code: str) -> Dict[str, Any]:
    if self.current_model_name == "baseline" or self.current_model is None:
      num_lines = code.count("\n") + (0 if code.endswith("\n") else 1 if code else 0)
      num_chars = len(code)
      return {
        "score": num_chars,
        "features": {"num_lines": num_lines, "num_chars": num_chars},
        "note": "baseline placeholder; replace with Assignment2 model inference",
      }
    try:
      y = self.current_model.predict([code])[0]
      proba = None
      if hasattr(self.current_model, "predict_proba"):
        proba = self.current_model.predict_proba([code])[0]
      resp = self._format_prediction(y, proba)
      resp.update({
        "features": {"code_length": len(code), "num_lines": code.count("\n") + (0 if code.endswith("\n") else 1 if code else 0)}
      })
      return resp
    except Exception as e:
      return {"error": f"Prediction failed: {str(e)}", "model_used": self.current_model_name, "fallback": "baseline"}
