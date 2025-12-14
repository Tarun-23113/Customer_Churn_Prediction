"""
API routes and endpoints
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from functools import lru_cache

from models import (
    PredictionRequest, PredictionResponse, HealthResponse, 
    ModelsResponse, FeatureImportanceResponse
)
from ml_service import ml_service
from config import settings

router = APIRouter()

@lru_cache(maxsize=1)
def get_api_info():
    """Cached API info"""
    return {
        "status": "running",
        "models": ml_service.get_model_count(),
        "version": settings.API_VERSION
    }

@router.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return get_api_info()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy", 
        models=ml_service.get_model_count()
    )

@router.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """Get available models with metadata"""
    return ModelsResponse(
        available_models=ml_service.get_model_list(),
        total=ml_service.get_model_count(),
        metadata=ml_service.model_metadata
    )

@router.get("/feature-ranges")
async def get_feature_ranges():
    """Get feature ranges for frontend"""
    if not ml_service.feature_ranges:
        raise HTTPException(status_code=500, detail="Feature ranges not loaded")
    
    return ml_service.feature_ranges

@router.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_churn(model_name: str, request: PredictionRequest):
    """Make churn prediction"""
    
    if model_name not in ml_service.get_model_list():
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Convert to numpy array
        features = np.array([[
            request.price,
            request.freight_value,
            request.payment_installments,
            request.delivery_diff_than_estimated,
            request.reviewed_days,
            request.customer_state_enc,
            request.product_category_name_enc,
            request.payment_type_enc
        ]])
        
        # Make prediction
        probability, prediction, confidence = ml_service.predict(model_name, features)
        
        return PredictionResponse(
            model_name=model_name,
            churn_probability=round(probability, 4),
            prediction=prediction,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/feature-importance/{model_name}", response_model=FeatureImportanceResponse)
async def get_feature_importance(model_name: str):
    """Get feature importance for model"""
    
    if model_name not in ml_service.get_model_list():
        raise HTTPException(status_code=404, detail="Model not found")
    
    result = ml_service.get_feature_importance(model_name)
    if result is None:
        raise HTTPException(status_code=400, detail="Feature importance not available")
    
    return FeatureImportanceResponse(
        model_name=model_name,
        **result
    )