from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from config import settings
from ml_service import ml_service
from api import router

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url=None
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize ML service on startup"""
    print("🚀 Starting ML API...")

    loaded_count, failed_models = ml_service.load_model()

    ml_service.load_feature_ranges()

    if failed_models:
        print(f"\n⚠️  Failed to load {len(failed_models)} model(s):")
        for name, error in failed_models:
            print(f"   - {name}: {error}")

    print(f"\n🎯 Final status: {loaded_count}/1 models loaded")
    print(f"📋 Available models: {ml_service.get_model_list()}")
    print(f"🚀 API ready with {loaded_count} models (Memory optimized)")
