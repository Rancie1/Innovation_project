import os
import pickle
from typing import List, Dict, Any
from model_loader import ModelLoader

class ModelService:
  def __init__(self) -> None:
    self._available: List[str] = ["logistic_regression", "random_forest", "model1_logreg", "model1_random_forest"]
    self.current_model_name: str | None = None
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
      dynamic: List[str] = []
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
        self._available = sorted(set(dynamic), key=str)
        # Auto-select first available model
        self.select_model(self._available[0])
        print(f"Loaded pre-trained models: {self._available}; selected '{self.current_model_name}'")
      else:
        print("No pre-trained models found. Please train models before predicting.")
    except Exception as e:
      print(f"Could not load pre-trained models: {e}. Please train models before predicting.")

  def available_models(self) -> List[str]:
    return list(self._available)

  def select_model(self, model_name: str) -> None:
    if model_name not in self._available:
      raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(self._available)}")
    
    self.current_model_name = model_name
    
    # Load the selected model
    try:
      with open(f"models/{model_name}.pkl", "rb") as f:
        self.current_model = pickle.load(f)
      print(f"Loaded model: {model_name}")
    except Exception as e:
      print(f"Could not load model {model_name}: {e}")
      self.current_model = None

  def _py_int(self, value: Any) -> Any:
    try:
      return int(value)
    except Exception:
      return value

  def _format_prediction(self, prediction: Any, proba: List[float] | None) -> Dict[str, Any]:
    # Normalize proba to a plain Python list if present
    proba_list: List[float] = []
    if proba is not None:
      try:
        proba_list = [float(x) for x in list(proba)]
      except Exception:
        proba_list = []

    # Model-2 style labels
    if self.current_model_name in ("logistic_regression", "random_forest") and self.target_names:
      idx = self._py_int(prediction)
      predicted_category = self.target_names[idx] if isinstance(idx, int) and 0 <= idx < len(self.target_names) else str(prediction)
      return {
        "predicted_category": predicted_category,
        "confidence": (max(proba_list) if proba_list else None),
        "all_probabilities": {name: p for name, p in zip(self.target_names, proba_list)},
        "model_used": self.current_model_name,
      }
    # Model-1 binary labels
    if self.current_model_name in ("model1_logreg", "model1_random_forest") and self.model1_classes is not None:
      py_label = self._py_int(prediction)
      
      # Map numeric labels to human-readable names
      label_map = {0: "Safe", 1: "Unsafe"}
      label_name = label_map.get(py_label, str(prediction))
      
      # Create all_probabilities dict for binary classification with readable names
      all_probabilities = {}
      if proba_list and len(proba_list) == len(self.model1_classes):
        for i, class_name in enumerate(self.model1_classes):
          # Map numeric class to readable name
          class_key = label_map.get(self._py_int(class_name), str(class_name))
          all_probabilities[class_key] = proba_list[i]
      
      # Map classes list to readable names
      readable_classes = [label_map.get(self._py_int(c), str(c)) for c in self.model1_classes]
      
      return {
        "predicted_label": py_label if isinstance(py_label, int) else str(py_label),
        "predicted_label_name": label_name,
        "classes": readable_classes,
        "confidence": (max(proba_list) if proba_list else None),
        "all_probabilities": all_probabilities,
        "model_used": self.current_model_name,
      }
    # Fallback generic
    py_pred = self._py_int(prediction)
    return {
      "predicted": py_pred if isinstance(py_pred, int) else str(py_pred),
      "confidence": (max(proba_list) if proba_list else None),
      "model_used": self.current_model_name,
    }

  def predict(self, code: str) -> Dict[str, Any]:
    if self.current_model is None or self.current_model_name is None:
      return {"error": "No model is loaded. Please train/select a model.", "model_used": None}
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
      return {"error": f"Prediction failed: {str(e)}", "model_used": self.current_model_name}
