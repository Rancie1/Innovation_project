import os
import pickle
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_model1"
MODELS_DIR = BASE_DIR / "models"

class Model1Loader:
  def __init__(self, train_csv: Path = DATA_DIR / "train.csv", test_csv: Path = DATA_DIR / "test.csv") -> None:
    self.train_csv = Path(train_csv)
    self.test_csv = Path(test_csv)

  def load_data(self) -> Dict[str, pd.DataFrame]:
    try:
      train_df = pd.read_csv(self.train_csv)
      test_df = pd.read_csv(self.test_csv)
    except FileNotFoundError:
      # Fallback to parquet if csvs missing
      train_parquet = self.train_csv.with_suffix('.parquet')
      test_parquet = self.test_csv.with_suffix('.parquet')
      if not train_parquet.exists() or not test_parquet.exists():
        raise FileNotFoundError(f"Could not find CSVs at {self.train_csv} / {self.test_csv} or Parquet at {train_parquet} / {test_parquet}")
      train_df = pd.read_parquet(train_parquet)
      test_df = pd.read_parquet(test_parquet)
    # Clean
    drop_cols = ["unique_id", "__index_level_0__"]
    def clean(df: pd.DataFrame) -> pd.DataFrame:
      df = df.drop(columns=drop_cols, errors="ignore")
      df = df.dropna(subset=["code"])  # ensure code present
      df = df.drop_duplicates(subset=["code"])  # dedupe by code
      return df
    return {"train": clean(train_df), "test": clean(test_df)}

  def build_logreg_pipeline(self) -> Pipeline:
    return Pipeline([
      ("tfidf", TfidfVectorizer(max_features=5000)),
      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

  def build_rf_pipeline(self) -> Pipeline:
    return Pipeline([
      ("tfidf", TfidfVectorizer(max_features=5000)),
      ("clf", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])

  def train_and_save(self, out_dir: Path = MODELS_DIR) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = self.load_data()
    train_df = data["train"]
    X = train_df["code"]
    y = train_df["target"]

    logreg = self.build_logreg_pipeline()
    rf = self.build_rf_pipeline()

    logreg.fit(X, y)
    rf.fit(X, y)

    models = {
      "model1_logreg": logreg,
      "model1_random_forest": rf,
    }

    for name, model in models.items():
      with open(out_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save class labels
    classes = sorted(list(pd.Series(y).unique()))
    with open(out_dir / "model1_classes.pkl", "wb") as f:
      pickle.dump(classes, f)

    return models

if __name__ == "__main__":
  loader = Model1Loader()
  models = loader.train_and_save()
  print("Model-1 training complete.")
