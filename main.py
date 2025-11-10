import streamlit as st
from pages import (
    project_overview,
    data_overview,
    churn_definition,
    data_quality,
    cleaning_steps,
    eda_insights,
    data_quality_after,
    feature_engineering,
    model_evaluation,
    model_interpretability,
    interactive_prediction,
    conclusion
)

# ---------------------------------------------------------
# 🔧 Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 🎨 Custom CSS Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {background-color: #0b1220; color: #e6eef6;}
    .stSidebar {background-color:#07111a;}
    h1, h2, h3, h4, h5, h6 {color: #e6eef6;}
    .stButton>button {
        background-color:#0f1720;
        color:#e6eef6;
        border:1px solid #233044;
        border-radius: 8px;
        padding: 0.4rem 1rem;
    }
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 🧠 Caching Heavy Objects
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_cached_data():
    import pandas as pd
    try:
        return pd.read_csv("data/final_cleaned_df.csv")
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    import joblib
    models = {}
    try:
        models["Logistic Regression"] = joblib.load("models/logistic_regression_grid.pkl")
        models["Random Forest"] = joblib.load("models/random_forest_grid.pkl")
        models["Gradient Boosting"] = joblib.load("models/gradient_boosting_grid.pkl")
        models["XGBoost"] = joblib.load("models/xgboost_grid.pkl")
    except Exception as e:
        st.warning(f"⚠️ Model loading issue: {e}")
    return models

# Preload everything when app starts
df_cached = load_cached_data()
models_cached = load_models()

# ---------------------------------------------------------
# 🧭 Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Interactive Prediction"])

# ---------------------------------------------------------
# 🏠 HOME PAGE (Load All Pages Sequentially)
# ---------------------------------------------------------
if page == "🏠 Home":
    st.title("E-Commerce Customer Churn Prediction Journey 🚀")
    st.caption("A complete walkthrough from data collection → EDA → modeling → deployment")

    with st.spinner("Loading complete project..."):
        st.markdown("#### 1️⃣ Project Overview")
        project_overview.app()

        st.markdown("---")
        st.markdown("#### 2️⃣ Data Overview")
        data_overview.app(df_cached)

        st.markdown("---")
        st.markdown("#### 3️⃣ Data Quality (Before Cleaning)")
        data_quality.app()

        st.markdown("---")
        st.markdown("#### 4️⃣ Data Cleaning Steps")
        cleaning_steps.app()\

        st.markdown("---")
        st.markdown("#### 5️⃣ Churn Definition")
        churn_definition.app()

        st.markdown("---")
        st.markdown("#### 6️⃣ EDA & Insights")
        eda_insights.app()

        st.markdown("---")
        st.markdown("#### 7️⃣ Data Quality (After Cleaning)")
        data_quality_after.app()

        st.markdown("---")
        st.markdown("#### 8️⃣ Feature Engineering")
        feature_engineering.app()

        st.markdown("---")
        st.markdown("#### 9️⃣ Model Training & Evaluation")
        model_evaluation.app()

        st.markdown("---")
        st.markdown("#### 🔟 Model Interpretation")
        model_interpretability.app()

        st.markdown("---")
        st.markdown("#### Conclusion")
        conclusion.app()

# ---------------------------------------------------------
# 🤖 INTERACTIVE PREDICTION PAGE
# ---------------------------------------------------------
elif page == "🤖 Interactive Prediction":
    st.title("Interactive Churn Prediction Tool 💡")
    interactive_prediction.app(models_cached)

