import os
import pickle
from typing import List, Dict, Any
from model_loader import ModelLoader

class ModelService:
  def __init__(self) -> None:
    self._available: List[str] = ["baseline", "logistic_regression", "random_forest"]
    self.current_model_name: str = "baseline"
    self.current_model = None
    self.label_encoder = None
    self.target_names = []
    self.model_loader = ModelLoader()
    
    # Try to load pre-trained models
    self._load_models()

  def _load_models(self):
    """Load pre-trained models if they exist."""
    try:
      models = self.model_loader.load_trained_models()
      if models:
        self._available = ["baseline"] + list(models.keys())
        self.label_encoder = self.model_loader.label_encoder
        self.target_names = self.model_loader.target_names
        print(f"Loaded {len(models)} pre-trained models: {list(models.keys())}")
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

  def predict(self, code: str) -> Dict[str, Any]:
    if self.current_model_name == "baseline" or self.current_model is None:
      # Placeholder baseline: returns simple features of the snippet
      num_lines = code.count("\n") + (0 if code.endswith("\n") else 1 if code else 0)
      num_chars = len(code)
      return {
        "score": num_chars,  # e.g., use length as a dummy score
        "features": {
          "num_lines": num_lines,
          "num_chars": num_chars,
        },
        "note": "baseline placeholder; replace with Assignment2 model inference",
      }
    
    try:
      # Use the trained model for prediction
      prediction = self.current_model.predict([code])[0]
      prediction_proba = self.current_model.predict_proba([code])[0]
      
      # Get the predicted category name
      if self.label_encoder and self.target_names:
        predicted_category = self.target_names[prediction]
        confidence = max(prediction_proba)
      else:
        predicted_category = f"Category_{prediction}"
        confidence = max(prediction_proba)
      
      return {
        "predicted_category": predicted_category,
        "confidence": float(confidence),
        "all_probabilities": {
          name: float(prob) for name, prob in zip(self.target_names, prediction_proba)
        },
        "model_used": self.current_model_name,
        "features": {
          "code_length": len(code),
          "num_lines": code.count("\n") + 1,
        }
      }
    except Exception as e:
      return {
        "error": f"Prediction failed: {str(e)}",
        "model_used": self.current_model_name,
        "fallback": "baseline"
      }
