import os

class FrontendConfig:
    # API Configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    
    # Cache Settings
    API_CACHE_TTL = 300
    FEATURE_CACHE_TTL = 600
    
    # UI Settings
    PAGE_TITLE = "Customer Churn Prediction App"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    SIDEBAR_STATE = "collapsed"
    
    # Request Settings
    REQUEST_TIMEOUT = 10
    HEALTH_CHECK_TIMEOUT = 3

config = FrontendConfig()