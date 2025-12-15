# 🚀 Customer Churn Prediction – End-to-End ML System

An end-to-end machine learning application that predicts customer churn for a Brazilian e-commerce platform using transactional data. The project covers the complete ML lifecycle: data preprocessing, feature engineering, model training and evaluation under class imbalance, and deployment using a FastAPI backend with a Streamlit frontend for real-time inference and insights.

---

## 📌 Problem Statement

Customer churn is a critical business problem for e-commerce platforms, as retaining existing customers is often more cost-effective than acquiring new ones.  
This project aims to identify customers at risk of churning based on their purchasing behavior, delivery experience, payment patterns, and review dynamics.

Since the dataset does not provide explicit churn labels, churn is defined using an inactivity-based proxy.

---

## 📊 Dataset Overview

- **Source**: Olist Brazilian E-commerce Dataset - https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
- **Time Period**: 2016–2018  
- **Nature**: Transactional, normalized relational data  

### Tables Used
- Customers  
- Orders  
- Order Items  
- Payments  
- Reviews  
- Products  
- Sellers  

The raw data is order-level, while churn is a customer-level concept, requiring careful aggregation and time-aware feature engineering.

---

## 🧠 Churn Definition (Proxy)

A customer is labeled as **churned (1)** if:

`Days since last purchase > 90 days`

Otherwise, the customer is labeled as **non-churned (0)**.

⚠️ **Note**: This is a heuristic proxy due to the absence of ground-truth churn labels. In production, this threshold should be validated using domain knowledge or sensitivity analysis.

---

## 🛠️ Data Processing & Feature Engineering

### Data Cleaning
- Retained only delivered orders to ensure availability of delivery and review signals  
- Handled missing values strategically:
  - Missing reviews preserved using sentinel values  
  - Product metadata filled using median or “Unknown” where appropriate  
- Removed redundant or post-engineering columns  

### Feature Engineering
Key behavioral and operational features:
- Delivery delay vs estimated delivery  
- Review timing after delivery  
- Freight value  
- Payment installments  
- Product category frequency  
- Customer geography  

### Categorical Encoding
- **Target Encoding**: `customer_state`  
  *(applied after train–test split to avoid leakage)*  
- **Frequency Encoding**: `product_category_name`  
- **Label Encoding**: `payment_type`  
  *(used only for tree-based models)*  

---

## 📈 Exploratory Data Analysis (EDA)

EDA was used to generate hypotheses, not final conclusions.

### Key Insights
- Delivery experience had a stronger impact on churn than price  
- Customers who reviewed sooner were less likely to churn  
- High freight costs correlated with increased churn risk  
- Non-linear relationships suggested tree-based models  

All insights were later validated using model performance.

---

## 🤖 Modeling Strategy

### Objective
Maximize the detection of churned customers in an imbalanced classification setting, prioritizing **Recall** and **ROC-AUC** over raw accuracy.

### Models Trained
- Logistic Regression (baseline & interpretability)  
- Random Forest  
- Gradient Boosting  
- XGBoost (primary model)  

### Imbalance Handling
- `class_weight='balanced'` for classical models  
- `scale_pos_weight` for XGBoost  

### Evaluation Metrics
- ROC-AUC  
- Recall  
- Precision  
- Confusion Matrix  

Accuracy was not prioritized due to class imbalance.

---

## 🏆 Model Selection

- Tree-based models outperformed linear models due to non-linear feature interactions  
- XGBoost achieved the best trade-off between recall and ROC-AUC  
- Train vs test metrics were compared to detect overfitting  

All trained models were serialized using `joblib`.

---

## 🧩 System Architecture
```
                ┌────────────────────────┐
                │       ML Training      │
                │     (Google Colab)     │
                │------------------------│
                │ • Data Cleaning        │
                │ • Data Pre-processing  │
                │ • EDA                  │
                │ • Feature Engineering  │
                │ • Model Training       │
                │ • Model Evaluation     │
                │ • Hyperparameter Tuning│
                └───────────┬────────────┘
                            │
                Saved Artifacts (.joblib)
                            │
        ┌───────────────────▼───────────────────┐
        │          Model Artifacts Layer        │
        │---------------------------------------│
        │ • XGBoost Model                       │
        │ • Random Forest                       │
        │ • Gradient Boosting                   │
        │ • Logistic Regression                 │
        └───────────────────┬───────────────────┘
                            │
                Loaded once at API startup
                            │
┌───────────────────────────▼───────────────────────────┐
│                    FastAPI Backend                    │
│-------------------------------------------------------│
│ • API Routes (/predict, /models, /health)             │
│ • Input Validation (Pydantic)                         │
│ • Feature Preprocessing                               │
│ • Model Selection & Inference                         │
│ • Feature Importance                                  │
│                                                       │
│     In-memory inference, stateless API                │
└───────────────────────────┬───────────────────────────┘
                            │
                     REST API Calls
                            │
┌───────────────────────────▼───────────────────────────┐
│                 Streamlit Frontend                    │
│-------------------------------------------------------│
│ • User Inputs (sliders, dropdowns)                    │
│ • Calls FastAPI                                       │
│ • Displays Predictions & Insights                     │
│                                                       │
│   No ML logic, no models, UI only                     │
└───────────────────────────────────────────────────────┘

- Frontend handles user input and visualization  
- Backend performs validation, preprocessing, and inference  
- Models are loaded once at application startup for low latency  
```
---

## 🔌 API Endpoints

- `GET /health` – Service health check  
- `GET /models` – List available models  
- `POST /predict/{model_name}` – Generate churn predictions  
- `GET /feature-importance/{model_name}` – Model interpretability  

---

## ⚡ Performance

- **Inference latency**: < 100 ms  
- **Memory usage**: < 512 MB  
- **Deployment**: Local development setup  
- **Scalability**: Horizontal scaling supported via API separation  

---

## 🖥️ Technologies Used

### Backend
- FastAPI  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  

### Frontend
- Streamlit  
- Seaborn  
- Requests  

---

## 🔮 Future Improvements

- Time-based cross-validation for more realistic evaluation  
- Probability calibration (Platt scaling / isotonic regression)  
- SHAP-based explainability  
- Drift detection and monitoring  
- Batch inference pipeline  
- Dockerized deployment  

---

## 🎯 Key Takeaways

- Demonstrates end-to-end ML Development  
- Focuses on business-aligned evaluation  
- Handled real-world constraints (no labels, imbalance, noisy data)  
- Clean separation between ML, API, and UI layers  