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
        "XGBoost": "saved_models/xgboost_grid.pkl"
    }

    # Feature Configuration
    FEATURE_NAMES: List[str] = [
        "price", "freight_value", "payment_installments",
        "delivery_diff_than_estimated", "reviewed_days",
        "customer_state_enc", "product_category_name_enc", "payment_type_enc"
    ]

    # Model Performance Results
    MODEL_PERFORMANCE: Dict[str, Dict] = {
        'XGBoost': {
            'test': {
                'roc_auc': 0.8826,
                'accuracy': 0.8614,
                'precision': 0.8683,
                'recall': 0.9753,
                'confusion_matrix': [[1816, 2749], [459, 18120]]
            },
            'train': {
                'roc_auc': 0.9003,
                'accuracy': 0.8698,
                'precision': 0.8729,
                'recall': 0.9805,
                'confusion_matrix': [[7654, 10606], [1446, 72870]]
            }
        },
        'Gradient Boosting': {
            'test': {
                'roc_auc': 0.8775,
                'accuracy': 0.8618,
                'precision': 0.8657,
                'recall': 0.9799,
                'confusion_matrix': [[1740, 2825], [374, 18205]]
            },
            'train': {
                'roc_auc': 0.8925,
                'accuracy': 0.8689,
                'precision': 0.8689,
                'recall': 0.9854,
                'confusion_matrix': [[7208, 11052], [1088, 73228]]
            }
        },
        'Random Forest': {
            'test': {
                'roc_auc': 0.8285,
                'accuracy': 0.7466,
                'precision': 0.9183,
                'recall': 0.7512,
                'confusion_matrix': [[3323, 1242], [4623, 13956]]
            },
            'train': {
                'roc_auc': 0.9999,
                'accuracy': 0.9996,
                'precision': 0.9999,
                'recall': 0.9996,
                'confusion_matrix': [[18251, 9], [27, 74289]]
            }
        },
        'Logistic Regression': {
            'test': {
                'roc_auc': 0.5791,
                'accuracy': 0.5463,
                'precision': 0.8265,
                'recall': 0.5504,
                'confusion_matrix': [[2418, 2147], [8354, 10225]]
            },
            'train': {
                'roc_auc': 0.5768,
                'accuracy': 0.5461,
                'precision': 0.8269,
                'recall': 0.5497,
                'confusion_matrix': [[9709, 8551], [33465, 40851]]
            }
        }
    }

    # Performance Settings
    CACHE_TTL = 300
    MAX_CACHE_SIZE = 32
    SAMPLE_SIZE = 1000 

    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    PORT = int(os.getenv("PORT", 8000))

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"


settings = Settings()
