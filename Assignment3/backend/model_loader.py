import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Dict, Any, List

class ModelLoader:
    """Loads and trains Assignment2 models for Assignment3 backend."""
    
    def __init__(self, data_path: str = "data/basic_data_3.cleaned.jsonl"):
        self.data_path = data_path
        self.models = {}
        self.label_encoder = None
        self.target_names = []
        
    def load_data(self) -> pd.DataFrame:
        """Load the cleaned data from Assignment2."""
        try:
            return pd.read_json(self.data_path, lines=True)
        except FileNotFoundError:
            print(f"Error: Data file not found at '{self.data_path}'")
            return pd.DataFrame()
    
    def prepare_data(self, df: pd.DataFrame):
        """Prepare features and labels for training."""
        X = df['code_snippet']
        Y = df['cwe_category']
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        Y_encoded = self.label_encoder.fit_transform(Y)
        self.target_names = self.label_encoder.classes_.tolist()
        
        return X, Y_encoded
    
    def train_logistic_regression(self, X, Y_encoded):
        """Train Logistic Regression model from Assignment2."""
        lr_pipeline = Pipeline([
            ('vect', CountVectorizer(token_pattern=r'\w{1,}', ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(random_state=42, solver='liblinear', max_iter=5000, C=10)),
        ])
        
        lr_pipeline.fit(X, Y_encoded)
        return lr_pipeline
    
    def train_random_forest(self, X, Y_encoded):
        """Train Random Forest model from Assignment2."""
        rf_pipeline = Pipeline([
            ('vect', CountVectorizer(token_pattern=r'\w{1,}', ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        
        rf_pipeline.fit(X, Y_encoded)
        return rf_pipeline
    
    def save_models(self, models: Dict[str, Any], output_dir: str = "models"):
        """Save trained models and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in models.items():
            with open(f"{output_dir}/{name}.pkl", "wb") as f:
                pickle.dump(model, f)
        
        # Save label encoder and target names
        with open(f"{output_dir}/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
        
        with open(f"{output_dir}/target_names.pkl", "wb") as f:
            pickle.dump(self.target_names, f)
        
        print(f"Models saved to {output_dir}/")
    
    def load_trained_models(self, model_dir: str = "models") -> Dict[str, Any]:
        """Load pre-trained models from disk."""
        models = {}
        
        # Load models
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl') and model_file not in ['label_encoder.pkl', 'target_names.pkl']:
                model_name = model_file.replace('.pkl', '')
                with open(f"{model_dir}/{model_file}", "rb") as f:
                    models[model_name] = pickle.load(f)
        
        # Load label encoder and target names
        if os.path.exists(f"{model_dir}/label_encoder.pkl"):
            with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
                self.label_encoder = pickle.load(f)
        
        if os.path.exists(f"{model_dir}/target_names.pkl"):
            with open(f"{model_dir}/target_names.pkl", "rb") as f:
                self.target_names = pickle.load(f)
        
        return models
    
    def train_all_models(self):
        """Train all Assignment2 models and save them."""
        print("Loading data...")
        df = self.load_data()
        if df.empty:
            return {}
        
        print("Preparing data...")
        X, Y_encoded = self.prepare_data(df)
        
        print("Training Logistic Regression...")
        lr_model = self.train_logistic_regression(X, Y_encoded)
        
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X, Y_encoded)
        
        models = {
            "logistic_regression": lr_model,
            "random_forest": rf_model
        }
        
        print("Saving models...")
        self.save_models(models)
        
        return models

if __name__ == "__main__":
    loader = ModelLoader()
    models = loader.train_all_models()
    print("Training complete!")
