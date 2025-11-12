import pandas as pd
import joblib
import os
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

def load_csv(path):
    return pd.read_csv(path)

def load_model(path):
    return joblib.load(path)

def load_image(path):
    try:
        return Image.open(path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def metrics_report(y_true, y_pred, y_prob):
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }

def evaluate_and_plot(model_name, model, x_train, x_test, y_train, y_test):
    y_train_pred = model.predict(x_train)
    y_train_prob = model.predict_proba(x_train)[:,1]
    y_test_pred = model.predict(x_test)
    y_test_prob = model.predict_proba(x_test)[:,1]
    train_metrics = metrics_report(y_train, y_train_pred, y_train_prob)
    test_metrics = metrics_report(y_test, y_test_pred, y_test_prob)
    return train_metrics, test_metrics
