import numpy as np
from fastapi import APIRouter, HTTPException

from models import PredictionRequest, PredictionResponse, HealthResponse
from ml_service import ml_service
from config import settings

router = APIRouter()


@router.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "version": settings.API_VERSION,
        "model": "XGBoost"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", models=1)


@router.get("/model-performance")
async def get_model_performance():
    """Get model performance metrics for all models"""
    return settings.MODEL_PERFORMANCE


@router.post("/predict/XGBoost", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """Make churn prediction using XGBoost"""
    try:
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

        probability, prediction, confidence = ml_service.predict(
            "XGBoost", features)

        return PredictionResponse(
            model_name="XGBoost",
            churn_probability=round(probability, 4),
            prediction=prediction,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

@router.get("/feature-importance/XGBoost")
async def get_feature_importance():
    """Get feature importance for XGBoost"""
    result = ml_service.get_feature_importance("XGBoost")
    if result is None:
        raise HTTPException(
            status_code=400, detail="Feature importance not available")

    return {
        "model_name": "XGBoost",
        **result
    }
