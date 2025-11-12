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

st.set_page_config(
    page_title="E-Commerce Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp {background-color: #ffffff; color: #1f2937;}
    .stSidebar {background-color: #f8f9fa;}
    h1, h2, h3, h4, h5, h6 {color: #1f2937;}
    .stButton>button {
        background-color: #ffffff;
        color: #1f2937;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.45rem 1rem;
    }
    .stButton>button:hover {
        background-color: #f3f4f6;
        border-color: #9ca3af;
    }
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🤖 Interactive Prediction"])

if page == "🏠 Home":
    st.markdown('<a id="top"></a>', unsafe_allow_html=True)
    st.title("E-Commerce Customer Churn Prediction Journey 🚀")
    st.caption("A complete end-to-end walkthrough: from data collection → EDA → modeling → deployment")

    with st.spinner("Loading complete project..."):
        st.markdown("#### 1️⃣ Project Overview")
        project_overview.app()

        st.markdown("---")
        st.markdown("#### 2️⃣ Data Overview")
        data_overview.app()

        st.markdown("---")
        st.markdown("#### 3️⃣ Data Quality (Before Cleaning)")
        data_quality.app()

        st.markdown("---")
        st.markdown("#### 4️⃣ Data Cleaning Steps")
        cleaning_steps.app()

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
        st.markdown("#### 🔟 Interactive Churn Prediction Tool 💡")
        interactive_prediction.app()

        st.markdown("---")
        st.markdown("#### 11️⃣ Conclusion & Business Takeaways")
        conclusion.app()

    st.markdown(
        """
        <div style="position:fixed; bottom:25px; right:30px; z-index:100;">
            <a href="#top">
                <button style="background-color:#3b82f6; color:white; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    ↑ Back to Top
                </button>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "🤖 Interactive Prediction":
    st.title("Interactive Churn Prediction Tool 💡")
    interactive_prediction.app()
