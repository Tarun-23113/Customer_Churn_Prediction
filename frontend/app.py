import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import os
from typing import Dict, List, Optional

# Optimized Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed for mobile optimization
)

# Custom CSS
st.markdown("""
<style>
.stApp {background-color: #ffffff; color: #1f2937;}
.stSidebar {background-color: #f8f9fa;}
h1, h2, h3, h4, h5, h6 {color: #1f2937;}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}
section[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache, no spinner
def check_api_health() -> bool:
    """Optimized API health check"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache
def get_available_models() -> List[str]:
    """Get available models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=3)
        if response.status_code == 200:
            return response.json()["available_models"]
        return []
    except:
        return []

@st.cache_data(ttl=600, show_spinner=False)  # 10 min cache for feature ranges
def get_feature_ranges() -> Optional[Dict]:
    """Get feature ranges for sliders"""
    try:
        response = requests.get(f"{API_BASE_URL}/feature-ranges", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(model_name: str, features: Dict) -> Optional[Dict]:
    """Optimized prediction with timeout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/{model_name}",
            json=features,
            timeout=10  # 10 second timeout
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_feature_importance(model_name):
    """Get feature importance from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/feature-importance/{model_name}")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def prediction_page():
    """Prediction page UI"""
    st.title("📊 Customer Churn Prediction")
    
    # Initialize session state
    if 'api_checked' not in st.session_state:
        st.session_state.api_checked = False
        st.session_state.api_healthy = False
        st.session_state.models = []
    
    # Check API health only once per session or when explicitly needed
    if not st.session_state.api_checked:
        with st.spinner("Connecting to API..."):
            st.session_state.api_healthy = check_api_health()
            if st.session_state.api_healthy:
                st.session_state.models = get_available_models()
            st.session_state.api_checked = True
    
    if not st.session_state.api_healthy:
        st.error("🚨 Backend API is not running. Please start the FastAPI server.")
        st.code("cd backend && uvicorn main:app --reload")
        if st.button("🔄 Retry Connection"):
            st.session_state.api_checked = False
            st.rerun()
        return
    
    # Get available models
    models = st.session_state.models
    if not models:
        st.error("No models available from API")
        if st.button("🔄 Refresh Models"):
            st.session_state.api_checked = False
            st.rerun()
        return
    
    # Sidebar controls
    st.sidebar.header("⚙️ Model Settings")
    selected_model = st.sidebar.selectbox("Choose Model", models)
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Feature inputs
    st.subheader("🧮 Customer Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price = st.slider("💰 Price", 0.0, 6735.0, 118.86, step=25.0)
        freight_value = st.slider("🚚 Freight Value", 0.0, 410.0, 20.10, step=5.0)
        payment_installments = st.slider("📅 Payment Installments", 0, 24, 3, step=1)
        delivery_diff = st.slider("📦 Delivery Difference (days)", -20, 150, 11, step=5)
    
    with col2:
        reviewed_days = st.slider("⭐ Reviewed Days", 0, 110, 0, step=5)
        customer_state = st.selectbox("📍 Customer State", 
                                    [0.696, 0.760, 0.782, 0.783, 0.788, 0.795, 0.796, 0.805, 0.810, 0.818, 0.820, 0.825, 0.867])
        product_category = st.selectbox("📦 Product Category", 
                                      [7, 14, 16, 24, 30, 31, 33, 37, 40, 45, 46, 59, 67, 72, 75, 87, 105, 119, 138, 140])
        payment_type = st.selectbox("💳 Payment Type", 
                                  [0, 1, 2, 3],
                                  format_func=lambda x: {0: "Credit Card", 1: "Boleto", 2: "Voucher", 3: "Debit Card"}[x])
    
    # Make prediction
    if st.button("🔮 Predict Churn", type="primary"):
        features = {
            "price": price,
            "freight_value": freight_value,
            "payment_installments": payment_installments,
            "delivery_diff_than_estimated": delivery_diff,
            "reviewed_days": reviewed_days,
            "customer_state_enc": customer_state,
            "product_category_name_enc": product_category,
            "payment_type_enc": payment_type
        }
        
        with st.spinner("Making prediction..."):
            result = make_prediction(selected_model, features)
        
        if result:
            st.markdown("---")
            st.subheader("🔍 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Churn Probability", f"{result['churn_probability']:.3f}")
            
            with col2:
                st.metric("Confidence", result['confidence'])
            
            with col3:
                st.metric("Model Used", result['model_name'])
            
            # Prediction result
            prob = result['churn_probability']
            if prob >= threshold:
                st.error("💔 **Predicted: Customer Will Churn**")
            else:
                st.success("💚 **Predicted: Customer Will Stay**")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Probability"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def insights_page():
    """Model insights and analytics page"""
    st.title("📈 Model Insights & Analytics")
    
    # Check API health
    if not check_api_health():
        st.error("🚨 Backend API is not running. Please start the FastAPI server.")
        return
    
    models = get_available_models()
    if not models:
        st.error("No models available from API")
        return
    
    # Model selection
    selected_model = st.selectbox("Select Model for Analysis", models)
    
    # Get feature importance
    importance_data = get_feature_importance(selected_model)
    
    if importance_data:
        st.subheader(f"🎯 Feature Importance - {selected_model}")
        
        # Feature importance chart
        features = list(importance_data['feature_importance'].keys())
        importance = list(importance_data['feature_importance'].values())
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title=f"Feature Importance - {selected_model}",
            labels={'x': 'Importance Score', 'y': 'Features'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top features
        st.subheader("🏆 Top 5 Most Important Features")
        top_features = importance_data['top_features']
        
        for i, (feature, score) in enumerate(top_features, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {feature.replace('_', ' ').title()}**")
            with col2:
                st.write(f"{score:.4f}")
        
        # Model comparison (placeholder for future)
        st.subheader("📊 Model Performance Comparison")
        st.info("Model performance metrics will be added here. This could include accuracy, precision, recall, and F1-scores for all models.")
        
        # Feature distribution insights (placeholder)
        st.subheader("📈 Feature Distribution Analysis")
        st.info("Feature distribution charts and correlation analysis will be added here.")

def main():
    """Main app with navigation"""
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose Page",
        ["🔮 Predictions", "📈 Insights"],
        label_visibility="collapsed"
    )
    
    # Route to pages
    if page == "🔮 Predictions":
        prediction_page()
    elif page == "📈 Insights":
        insights_page()

if __name__ == "__main__":
    main()