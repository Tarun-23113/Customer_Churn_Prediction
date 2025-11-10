import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title("Model Interpretability")
    try:
        X = pd.read_csv("data/x_test.csv")
    except Exception:
        st.error("x_test.csv missing")
        return
    models = {
        "Random Forest": "models/random_forest_grid.pkl",
        "Gradient Boosting": "models/gradient_boosting_grid.pkl",
        "XGBoost": "models/xgboost_grid.pkl",
        "Logistic Regression": "models/logistic_regression_grid.pkl"
    }
    selected = st.selectbox("Select model", list(models.keys()))
    try:
        model = joblib.load(models[selected])
    except Exception:
        st.error("Model file not found")
        return
    if selected == "Logistic Regression":
        try:
            coefs = model.named_steps['clf'].coef_[0]
            feats = X.columns.tolist()
            df_imp = pd.DataFrame({"feature": feats, "importance": coefs}).sort_values("importance", ascending=False)
        except Exception:
            st.error("Unable to extract coefficients")
            return
    else:
        try:
            feats = X.columns.tolist()
            imp = model.feature_importances_
            df_imp = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
        except Exception:
            st.error("Unable to extract feature importances")
            return
    st.dataframe(df_imp.head(20))
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=df_imp.head(20), x="importance", y="feature", ax=ax)
    st.pyplot(fig)
