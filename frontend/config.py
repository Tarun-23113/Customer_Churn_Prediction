"""
Frontend configuration
"""
import os

class FrontendConfig:
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Cache Settings
    API_CACHE_TTL = 300  # 5 minutes
    FEATURE_CACHE_TTL = 600  # 10 minutes
    
    # UI Settings
    PAGE_TITLE = "Churn Prediction"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    SIDEBAR_STATE = "collapsed"
    
    # Request Settings
    REQUEST_TIMEOUT = 10
    HEALTH_CHECK_TIMEOUT = 3

config = FrontendConfig()