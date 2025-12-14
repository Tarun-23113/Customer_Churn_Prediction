"""
Configuration settings for the ML API
"""
import os
from typing import Dict, List

class Settings:
    # API Configuration
    API_TITLE = "Customer Churn Prediction API"
    API_DESCRIPTION = "Optimized ML API for customer churn prediction"
    API_VERSION = "1.0.0"
    
    # Model Configuration
    MODEL_FILES: Dict[str, str] = {
        "XGBoost": "models/xgboost_grid.pkl",
        "Random Forest": "models/random_forest_grid.pkl", 
        "Gradient Boosting": "models/gradient_boosting_grid.pkl",
        "Logistic Regression": "models/logistic_regression_grid.pkl"
    }
    
    # Feature Configuration
    FEATURE_NAMES: List[str] = [
        "price", "freight_value", "payment_installments",
        "delivery_diff_than_estimated", "reviewed_days",
        "customer_state_enc", "product_category_name_enc", "payment_type_enc"
    ]
    
    # Performance Settings
    CACHE_TTL = 300  # 5 minutes
    MAX_CACHE_SIZE = 32
    SAMPLE_SIZE = 1000  # For feature stats
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    PORT = int(os.getenv("PORT", 8000))
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

settings = Settings()