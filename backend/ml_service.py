import os
import gc
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Dict, Optional, Tuple, Any, List
from config import settings


class MLService:
    """Service class for ML operations"""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.feature_ranges: Dict[str, Dict[str, float]] = {}
        self._script_dir = os.path.dirname(os.path.abspath(__file__))

    def load_model(self) -> Tuple[int, List[Tuple[str, str]]]:
        """Load ML model and return success"""
        loaded_count = 0
        failed_models: List[Tuple[str, str]] = []

        for name, path in settings.MODEL_FILES.items():
            full_path = os.path.join(self._script_dir, path)
            print(f"🔄 Loading {name} from {full_path}")

            if os.path.exists(full_path):
                try:
                    model = joblib.load(full_path)
                    self.models[name] = model

                    self.model_metadata[name] = {
                        "type": type(model).__name__,
                        "has_feature_importance": hasattr(model, "feature_importances_") or hasattr(model, "coef_")
                    }

                    loaded_count += 1
                    print(f"✅ Loaded {name} ({type(model).__name__})")
                    gc.collect()

                except Exception as e:
                    failed_models.append((name, str(e)))
                    print(f"❌ Failed to load {name}: {e}")
            else:
                failed_models.append((name, "File not found"))
                print(f"❌ Model file not found: {full_path}")

        return loaded_count, failed_models

    def load_feature_ranges(self) -> bool:
        """Load feature ranges for validation"""
        x_test_path = os.path.join(self._script_dir, "x_test.csv")

        if os.path.exists(x_test_path):
            try:
                sample_data = pd.read_csv(
                    x_test_path, index_col=0, nrows=settings.SAMPLE_SIZE)
                stats = sample_data.describe()

                self.feature_ranges = {
                    col: {
                        "min": float(stats.loc["min", col]),
                        "max": float(stats.loc["max", col]),
                        "mean": float(stats.loc["mean", col])
                    }
                    for col in sample_data.columns
                }

                del sample_data, stats
                gc.collect()
                print("✅ Loaded feature ranges")
                return True

            except Exception as e:
                print(f"❌ Failed to load feature ranges: {e}")
                return False

        print("❌ x_test.csv not found")
        return False

    def predict(self, model_name: str, features: np.ndarray) -> Tuple[float, int, str]:
        """Make prediction with given model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        probability = float(model.predict_proba(features)[0, 1])
        prediction = int(probability >= 0.5)

        if probability > 0.9:
            confidence = "High Probability that it will Churn"
        elif probability > 0.8:
            confidence = "Medium Probability that it will Churn"
        elif probability <= 0.1:
            confidence = "High Probability that it will not Churn"
        elif probability < 0.2:
            confidence = "Medium Probability that it will not Churn"
        else:
            confidence = "Low Confidence (Neutral)"

        return probability, prediction, confidence

    @lru_cache(maxsize=settings.MAX_CACHE_SIZE)
    def get_feature_importance(self, model_name: str) -> Optional[Dict]:
        """Get cached feature importance"""
        if model_name not in self.models:
            return None

        if not self.model_metadata.get(model_name, {}).get("has_feature_importance", False):
            return None

        model = self.models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None

        feature_importance = dict(
            zip(settings.FEATURE_NAMES, importance.tolist()))
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True)

        return {
            "feature_importance": dict(sorted_features),
            "top_features": sorted_features[:5]
        }

    def get_model_list(self) -> list:
        """Get list of available models"""
        return list(self.models.keys())

    def get_model_count(self) -> int:
        """Get number of loaded models"""
        return len(self.models)

ml_service = MLService()
