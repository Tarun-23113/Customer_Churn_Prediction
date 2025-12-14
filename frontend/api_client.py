"""
API client for backend communication
"""
import streamlit as st
import requests
from typing import Dict, List, Optional
from config import config

# Standalone cached functions (fix for Streamlit caching)

@st.cache_data(ttl=config.API_CACHE_TTL, show_spinner=False)
def check_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/health", 
            timeout=config.HEALTH_CHECK_TIMEOUT
        )
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=config.API_CACHE_TTL, show_spinner=False)
def get_models() -> List[str]:
    """Get available models"""
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/models", 
            timeout=config.HEALTH_CHECK_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()["available_models"]
        return []
    except:
        return []

@st.cache_data(ttl=config.FEATURE_CACHE_TTL, show_spinner=False)
def get_feature_ranges() -> Optional[Dict]:
    """Get feature ranges for sliders"""
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/feature-ranges", 
            timeout=config.HEALTH_CHECK_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict(model_name: str, features: Dict) -> Optional[Dict]:
    """Make prediction"""
    try:
        response = requests.post(
            f"{config.API_BASE_URL}/predict/{model_name}",
            json=features,
            timeout=config.REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Prediction failed. Please try again.")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except Exception:
        st.error("Connection error. Please check if the API is running.")
        return None

@st.cache_data(ttl=config.FEATURE_CACHE_TTL, show_spinner=False)
def get_feature_importance(model_name: str) -> Optional[Dict]:
    """Get feature importance"""
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/feature-importance/{model_name}",
            timeout=config.REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None