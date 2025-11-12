import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

def app():
    st.header("📊 Model Evaluation & Comparison")

    try:
        x_train = pd.read_csv("data/x_train.csv", index_col=0)
        x_test = pd.read_csv("data/x_test.csv", index_col=0)
        y_train = pd.read_csv("data/y_train.csv", index_col=0).values.ravel()
        y_test = pd.read_csv("data/y_test.csv", index_col=0).values.ravel()

        rf_model = joblib.load("models/random_forest_grid.pkl")
        gb_model = joblib.load("models/gradient_boosting_grid.pkl")
        xgb_model = joblib.load("models/xgboost_grid.pkl")
        lr_model = joblib.load("models/logistic_regression_grid.pkl")
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.warning("""
        **Model Loading Failed - Version Compatibility Issue**
        
        This error typically occurs when models were trained with a different version of numpy/scikit-learn.
        
        **Solutions:**
        1. Retrain your models with the current environment
        2. Downgrade numpy to match the version used during training
        3. Check your requirements.txt for version compatibility
        
        Run: `pip list | findstr "numpy scikit-learn"` to check current versions
        """)
        return

    models = {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "Gradient Boosting": gb_model,
        "Logistic Regression": lr_model
    }

    st.markdown("### 🧠 Evaluating All Models on Test and Train Data")

    results_test, results_train = [], []

    cols = st.columns(2)
    idx = 0

    for name, model in models.items():
        test_preds = model.predict(x_test)
        test_probs = model.predict_proba(x_test)[:, 1]
        train_preds = model.predict(x_train)
        train_probs = model.predict_proba(x_train)[:, 1]

        test_metrics = {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, test_probs),
            "Accuracy": accuracy_score(y_test, test_preds),
            "Precision": precision_score(y_test, test_preds),
            "Recall": recall_score(y_test, test_preds),
        }
        train_metrics = {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_train, train_probs),
            "Accuracy": accuracy_score(y_train, train_preds),
            "Precision": precision_score(y_train, train_preds),
            "Recall": recall_score(y_train, train_preds),
        }

        results_test.append(test_metrics)
        results_train.append(train_metrics)

        cm = confusion_matrix(y_test, test_preds)
        with cols[idx % 2]:
            st.markdown(f"#### {name} (Test Data)")
            st.dataframe(pd.DataFrame([test_metrics]).set_index("Model"))
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f"{name} - Confusion Matrix (Test)")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        idx += 1

    st.markdown("## 📈 Model Comparison (Test Results)")
    df_test = pd.DataFrame(results_test).sort_values("ROC-AUC", ascending=False)
    st.dataframe(df_test[["Model", "ROC-AUC", "Accuracy"]])

    fig, ax = plt.subplots(figsize=(8, 4))
    df_melted = df_test.melt(id_vars="Model", value_vars=["ROC-AUC", "Accuracy"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison (ROC-AUC & Accuracy) - Test Data")
    st.pyplot(fig)

    st.markdown("## 🧩 Model Comparison (Train Results)")
    df_train = pd.DataFrame(results_train).sort_values("ROC-AUC", ascending=False)
    st.dataframe(df_train[["Model", "ROC-AUC", "Accuracy"]])

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df_melted_train = df_train.melt(id_vars="Model", value_vars=["ROC-AUC", "Accuracy"], var_name="Metric", value_name="Score")
    sns.barplot(data=df_melted_train, x="Model", y="Score", hue="Metric", ax=ax2)
    ax2.set_ylim(0, 1)
    ax2.set_title("Model Performance Comparison (ROC-AUC & Accuracy) - Train Data")
    st.pyplot(fig2)
