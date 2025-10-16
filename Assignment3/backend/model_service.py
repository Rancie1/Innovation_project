from typing import List, Dict, Any

class ModelService:
  def __init__(self) -> None:
    self._available: List[str] = ["baseline"]
    self.current_model_name: str = "baseline"

  def available_models(self) -> List[str]:
    return list(self._available)

  def select_model(self, model_name: str) -> None:
    if model_name not in self._available:
      raise ValueError(f"Unknown model '{model_name}'. Available: {', '.join(self._available)}")
    self.current_model_name = model_name

  def predict(self, code: str) -> Dict[str, Any]:
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
