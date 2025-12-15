import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional

from config import config
import api_client

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE
)

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


def prediction_page():
    """Prediction page UI"""
    st.title("📊 Customer Churn Prediction")
    st.info("🏆 Using XGBoost - Our Best Performing Model (ROC-AUC: 0.8826)")

    # Initialize session state
    if 'api_checked' not in st.session_state:
        st.session_state.api_checked = False
        st.session_state.api_healthy = False

    # Check API health only once per session
    if not st.session_state.api_checked:
        with st.spinner("Connecting to API..."):
            st.session_state.api_healthy = api_client.check_health()
            st.session_state.api_checked = True

    if not st.session_state.api_healthy:
        st.error("🚨 Backend API is not running. Please start the FastAPI server.")
        st.code("cd backend && uvicorn main:app --reload")
        if st.button("🔄 Retry Connection"):
            st.session_state.api_checked = False
            st.rerun()
        return

    # Sidebar info
    st.sidebar.header("ℹ️ About the Model")
    st.sidebar.info(
        "🏆 **XGBoost Classifier**\n\n"
        "- ROC-AUC: 0.8826\n"
        "- Accuracy: 86.14%\n"
        "- Recall: 97.53%\n\n"
        "Uses 0.5 threshold for classification.")

    # Feature inputs
    st.subheader("🧮 Customer Features")

    col1, col2 = st.columns(2)

    with col1:
        price = st.slider("💰 Price", 0.0, 6735.0, 118.86, step=25.0)
        freight_value = st.slider(
            "🚚 Freight Value", 0.0, 410.0, 20.10, step=5.0)
        payment_installments = st.slider(
            "📅 Payment Installments", 0, 24, 3, step=1)
        delivery_diff = st.slider(
            "📦 Delivery Difference (days)", -20, 150, 11, step=5)

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
            result = api_client.predict("XGBoost", features)

        if result:
            st.markdown("---")
            st.subheader("🔍 Prediction Result")

            prediction = result['prediction']

            if prediction:
                st.error("**Customer Will Churn ( will not stay )**")
            else:
                st.success("**Customer Will Not Churn**")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Prediction Probability",
                          f"{result['churn_probability']:.3f}")

            with col2:
                st.metric("Confidence", result['confidence'])
            

def insights_page():
    """Model insights and analytics page"""
    st.title("📈 Model Insights & Analytics")

    # Check API health
    if not api_client.check_health():
        st.error("🚨 Backend API is not running. Please start the FastAPI server.")
        return

    # Get feature importance for XGBoost
    importance_data = api_client.get_feature_importance("XGBoost")

    if importance_data:
        st.subheader("🎯 Feature Importance - XGBoost (Best Model)")

        # Feature importance chart using seaborn
        features = list(importance_data['feature_importance'].keys())
        importance = list(importance_data['feature_importance'].values())

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create horizontal bar chart
        y_pos = range(len(features))
        bars = ax.barh(y_pos, importance, color=sns.color_palette(
            "viridis", len(features)))

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title(
            'Feature Importance - XGBoost', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Top features
        st.subheader("🏆 Top 5 Most Important Features")
        top_features = importance_data['top_features']

        for i, (feature, score) in enumerate(top_features, 1):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i}. {feature.replace('_', ' ').title()}**")
            with col2:
                st.write(f"{score:.4f}")

    st.markdown("---")

    # Model Performance Comparison
    st.subheader("📊 Model Performance Comparison")
    st.write("Comparing all models on both Training and Testing datasets to understand bias-variance tradeoff")

    # Fetch model performance data from API
    models_data = api_client.get_model_performance()

    if not models_data:
        st.error("Failed to load model performance data from backend.")
        return

    # 1. Metrics Comparison - Training vs Testing
    st.subheader("📈 Training vs Testing Performance")

    metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    metric_names = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]

        model_names = list(models_data.keys())
        train_scores = [models_data[m]['train'][metric] for m in model_names]
        test_scores = [models_data[m]['test'][metric] for m in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, train_scores, width,
                       label='Training', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_scores, width,
                       label='Testing', color='coral', alpha=0.8)

        ax.set_xlabel('Models', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} - Training vs Testing',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 2. Bias-Variance Analysis
    st.subheader("⚖️ Bias-Variance Tradeoff Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🔍 Overfitting Detection**")
        st.write("Gap between Training and Testing Performance:")

        fig, ax = plt.subplots(figsize=(10, 6))

        model_names = list(models_data.keys())
        overfit_scores = [models_data[m]['train']['accuracy'] - models_data[m]['test']['accuracy']
                          for m in model_names]

        colors = ['red' if score > 0.1 else 'orange' if score > 0.05 else 'green'
                  for score in overfit_scores]

        bars = ax.barh(model_names, overfit_scores, color=colors, alpha=0.7)
        ax.set_xlabel('Accuracy Gap (Train - Test)', fontsize=11)
        ax.set_title('Overfitting Indicator', fontsize=12, fontweight='bold')
        ax.axvline(0.05, color='orange', linestyle='--',
                   linewidth=1, label='Moderate Gap')
        ax.axvline(0.1, color='red', linestyle='--',
                   linewidth=1, label='High Gap')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, score) in enumerate(zip(bars, overfit_scores)):
            ax.text(score, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("🟢 **Green**: Good generalization\n🟠 **Orange**: Moderate overfitting\n🔴 **Red**: High overfitting")

    with col2:
        st.markdown("**📊 Model Consistency**")
        st.write("Standard deviation across metrics (lower is better):")

        consistency_scores = {}
        for model in model_names:
            test_metrics = [models_data[model]['test'][m] for m in metrics]
            consistency_scores[model] = np.std(test_metrics)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(list(consistency_scores.keys()), list(consistency_scores.values()),
                       color=sns.color_palette("coolwarm", len(consistency_scores)))
        ax.set_xlabel('Standard Deviation', fontsize=11)
        ax.set_title('Model Consistency Score', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info(
            "Lower standard deviation indicates more balanced performance across all metrics.")

    # 3. Confusion Matrix Comparison
    st.subheader("🔢 Confusion Matrix Comparison (Testing Set)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, model in enumerate(model_names):
        ax = axes[idx]
        cm = np.array(models_data[model]['test']['confusion_matrix'])

        # Plot confusion matrix using seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])

        ax.set_title(f'{model}\nAccuracy: {models_data[model]["test"]["accuracy"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # 4. Model Selection Recommendation
    st.subheader("🏆 Model Selection Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("**🥇 Best Overall: XGBoost**")
        st.write("- Highest ROC-AUC on test set (0.8826)")
        st.write("- Excellent balance of precision & recall")
        st.write("- Minimal overfitting (0.0084 gap)")
        st.write("- Good generalization capability")

    with col2:
        st.warning("**⚠️ Overfitting Alert: Random Forest**")
        st.write("- Perfect training score (0.9999 ROC-AUC)")
        st.write("- Large performance drop on test set")
        st.write("- Accuracy gap: 0.2530")
        st.write("- High variance, poor generalization")

    with col3:
        st.error("**❌ Poor Performance: Logistic Regression**")
        st.write("- Low ROC-AUC (0.5791)")
        st.write("- Barely better than random")
        st.write("- High bias, underfitting")
        st.write("- Not suitable for this task")


def main():
    """Main app with navigation"""

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose Page",
        ["🔮 Predictions", "📈 Insights"],
        label_visibility="collapsed"
    )

    if page == "🔮 Predictions":
        prediction_page()
    elif page == "📈 Insights":
        insights_page()


if __name__ == "__main__":
    main()
