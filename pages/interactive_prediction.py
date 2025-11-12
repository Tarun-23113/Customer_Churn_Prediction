import streamlit as st
import pandas as pd
import joblib
import numpy as np

def app():
    st.title("📊 Real-Time Customer Churn Prediction")

    try:
        sample = pd.read_csv("data/x_test.csv", index_col=0).head(1)
    except Exception:
        st.error("⚠️ Missing file: data/x_test.csv")
        return

    models = {
        "Logistic Regression": "models/logistic_regression_grid.pkl",
        "Random Forest": "models/random_forest_grid.pkl",
        "Gradient Boosting": "models/gradient_boosting_grid.pkl",
        "XGBoost": "models/xgboost_grid.pkl"
    }

    st.sidebar.header("⚙️ Model & Threshold Settings")
    selected = st.sidebar.selectbox("Choose Model", list(models.keys()))
    threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01)

    try:
        stats = pd.read_csv("data/describe.csv", index_col=0)
    except Exception:
        stats = sample.describe()

    st.subheader("🧮 Input Features")
    inputs = {}

    for col in sample.columns:
        if pd.api.types.is_numeric_dtype(sample[col]):
            min_val = float(stats.loc["min", col]) if "min" in stats.index else float(sample[col].min())
            max_val = float(stats.loc["max", col]) if "max" in stats.index else float(sample[col].max())
            mean_val = float(stats.loc["mean", col]) if "mean" in stats.index else float(sample[col].iloc[0])
            
            # Handle case where min == max (no variance in column)
            if min_val == max_val:
                inputs[col] = st.number_input(col, value=mean_val)
            else:
                inputs[col] = st.slider(col, min_val, max_val, mean_val)
        else:
            inputs[col] = st.text_input(col, str(sample[col].iloc[0]))

    X = pd.DataFrame([inputs])

    try:
        model = joblib.load(models[selected])
    except Exception:
        st.error(f"⚠️ Model file not found for {selected}")
        return

    try:
        prob = model.predict_proba(X)[:, 1][0]
        pred = int(prob >= threshold)
        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        st.metric("Churn Probability", f"{prob:.3f}")

        if pred == 1:
            st.error("💔 Predicted: **Customer Will Churn**")
        else:
            st.success("💚 Predicted: **Customer Will Stay**")

    except Exception as e:
        st.warning(f"Prediction failed: {e}")
