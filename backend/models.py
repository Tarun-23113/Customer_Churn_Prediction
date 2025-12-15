from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class PredictionRequest(BaseModel):
    """Request model for churn prediction"""
    price: float = Field(..., ge=0, le=10000, description="Product price")
    freight_value: float = Field(..., ge=0, le=500, description="Freight cost")
    payment_installments: int = Field(..., ge=0, le=24, description="Payment installments")
    delivery_diff_than_estimated: int = Field(..., ge=-50, le=200, description="Delivery difference in days")
    reviewed_days: int = Field(..., ge=0, le=150, description="Days since review")
    customer_state_enc: float = Field(..., ge=0, le=1, description="Encoded customer state")
    product_category_name_enc: int = Field(..., ge=0, le=15000, description="Encoded product category")
    payment_type_enc: int = Field(..., ge=0, le=3, description="Encoded payment type")

class PredictionResponse(BaseModel):
    """Response model for churn prediction"""
    model_name: str
    churn_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., ge=0, le=1)
    confidence: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models: int

class ModelsResponse(BaseModel):
    """Available models response"""
    available_models: List[str]
    total: int
    metadata: Dict[str, Dict]

class FeatureImportanceResponse(BaseModel):
    """Feature importance response"""
    model_name: str
    feature_importance: Dict[str, float]
    top_features: List[tuple]