from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
from typing import Dict, List, Optional
import os
import gc
from functools import lru_cache

# Optimized FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="Optimized ML API for customer churn prediction",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url=None  # Disable redoc to save memory
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Optimized CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Disable for better performance
    allow_methods=["GET", "POST"],  # Only needed methods
    allow_headers=["*"],
)

# Global variables - optimized
models: Dict[str, object] = {}
feature_ranges: Dict[str, Dict[str, float]] = {}
model_metadata: Dict[str, Dict] = {}

class PredictionRequest(BaseModel):
    price: float = Field(..., ge=0, le=10000, description="Product price")
    freight_value: float = Field(..., ge=0, le=500, description="Freight cost")
    payment_installments: int = Field(..., ge=0, le=24, description="Payment installments")
    delivery_diff_than_estimated: int = Field(..., ge=-50, le=200, description="Delivery difference in days")
    reviewed_days: int = Field(..., ge=0, le=150, description="Days since review")
    customer_state_enc: float = Field(..., ge=0, le=1, description="Encoded customer state")
    product_category_name_enc: int = Field(..., ge=0, le=15000, description="Encoded product category")
    payment_type_enc: int = Field(..., ge=0, le=3, description="Encoded payment type")

class PredictionResponse(BaseModel):
    model_name: str
    churn_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., ge=0, le=1)
    confidence: str

@app.on_event("startup")
async def load_models():
    """Optimized model loading with memory management"""
    global models, feature_ranges, model_metadata
    
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Prioritized model loading (load most accurate first)
        model_files = {
            "XGBoost": os.path.join(script_dir, "models/xgboost_grid.pkl"),
            "Random Forest": os.path.join(script_dir, "models/random_forest_grid.pkl"), 
            "Gradient Boosting": os.path.join(script_dir, "models/gradient_boosting_grid.pkl"),
            "Logistic Regression": os.path.join(script_dir, "models/logistic_regression_grid.pkl")
        }
        
        loaded_count = 0
        failed_models = []
        
        for name, path in model_files.items():
            print(f"🔄 Attempting to load {name} from {path}")
            
            if os.path.exists(path):
                try:
                    print(f"   📁 File exists, loading...")
                    model = joblib.load(path)
                    models[name] = model
                    
                    # Store model metadata for quick access
                    model_metadata[name] = {
                        "type": type(model).__name__,
                        "has_feature_importance": hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')
                    }
                    
                    loaded_count += 1
                    print(f"✅ Successfully loaded {name} ({type(model).__name__})")
                    
                    # Force garbage collection after each model
                    gc.collect()
                    
                except Exception as e:
                    failed_models.append((name, str(e)))
                    print(f"❌ Failed to load {name}: {e}")
                    print(f"   Error type: {type(e).__name__}")
            else:
                failed_models.append((name, "File not found"))
                print(f"❌ Model file not found: {path}")
        
        if failed_models:
            print(f"\n⚠️  Failed to load {len(failed_models)} models:")
            for name, error in failed_models:
                print(f"   - {name}: {error}")
        
        print(f"\n🎯 Final status: {loaded_count}/4 models loaded")
        print(f"📋 Available models: {list(models.keys())}")
        
        # Load minimal feature data (only ranges, not full dataset)
        x_test_path = os.path.join(script_dir, "x_test.csv")
        if os.path.exists(x_test_path):
            # Read only first 1000 rows for stats to save memory
            sample_data = pd.read_csv(x_test_path, index_col=0, nrows=1000)
            stats = sample_data.describe()
            
            # Store only essential ranges
            feature_ranges = {
                col: {
                    "min": float(stats.loc["min", col]),
                    "max": float(stats.loc["max", col]),
                    "mean": float(stats.loc["mean", col])
                }
                for col in sample_data.columns
            }
            
            # Clear sample data to free memory
            del sample_data, stats
            gc.collect()
            print("✅ Loaded feature ranges")
        
        print(f"🚀 API ready with {loaded_count} models (Memory optimized)")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@lru_cache(maxsize=1)
def get_api_info():
    """Cached API info"""
    return {
        "status": "running",
        "models": len(models),
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    return get_api_info()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": len(models)}

@app.get("/models")
async def get_available_models():
    """Get available models with metadata"""
    return {
        "available_models": list(models.keys()),
        "total": len(models),
        "metadata": model_metadata
    }

@app.get("/feature-ranges")
async def get_feature_ranges():
    """Get optimized feature ranges"""
    if not feature_ranges:
        raise HTTPException(status_code=500, detail="Feature ranges not loaded")
    
    return feature_ranges

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict_churn(model_name: str, request: PredictionRequest):
    """Optimized prediction endpoint"""
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Convert to numpy array directly (faster than DataFrame)
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
        
        # Fast prediction
        model = models[model_name]
        probability = float(model.predict_proba(features)[0, 1])
        prediction = int(probability >= 0.5)
        
        # Optimized confidence calculation
        confidence = "High" if probability > 0.75 or probability < 0.25 else "Medium" if probability > 0.6 or probability < 0.4 else "Low"
        
        return PredictionResponse(
            model_name=model_name,
            churn_probability=round(probability, 4),
            prediction=prediction,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

@lru_cache(maxsize=32)  # Cache feature importance results
def _get_cached_feature_importance(model_name: str):
    """Cached feature importance calculation"""
    model = models[model_name]
    
    feature_names = [
        "price", "freight_value", "payment_installments",
        "delivery_diff_than_estimated", "reviewed_days", 
        "customer_state_enc", "product_category_name_enc", "payment_type_enc"
    ]
    
    # Get importance based on model type
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None
    
    # Create sorted feature importance
    feature_importance = dict(zip(feature_names, importance.tolist()))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "feature_importance": dict(sorted_features),
        "top_features": sorted_features[:5]
    }

@app.get("/feature-importance/{model_name}")
async def get_feature_importance(model_name: str):
    """Optimized feature importance endpoint"""
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model_metadata.get(model_name, {}).get("has_feature_importance", False):
        raise HTTPException(status_code=400, detail="Feature importance not available")
    
    try:
        result = _get_cached_feature_importance(model_name)
        if result is None:
            raise HTTPException(status_code=400, detail="Feature importance not supported")
        
        return {"model_name": model_name, **result}
        
    except Exception:
        raise HTTPException(status_code=500, detail="Error getting feature importance")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)