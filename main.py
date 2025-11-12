import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Model Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Model Prediction")

# Add Google Colab link
st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="margin: 0;">📓 Full Implementation</h4>
    <p style="margin: 5px 0;">View the complete code and implementation in Google Colab:</p>
    <a href="https://colab.research.google.com/drive/1TcvYUN_2fiuv9FQ6JMkjBIHbOW1TzvzZ?usp=sharing" target="_blank" style="text-decoration: none;">
        <button style="background-color: #FF6F00; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
            Open in Google Colab
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

# Model selection
models = {
    "Random Forest": "models/random_forest_grid.pkl",
    "Gradient Boosting": "models/gradient_boosting_grid.pkl",
    "XGBoost": "models/xgboost_grid.pkl",
    "Logistic Regression": "models/logistic_regression_grid.pkl"
}

selected_model = st.selectbox("Select Model", list(models.keys()))

# Load model
try:
    model = joblib.load(models[selected_model])
    st.success(f"✓ {selected_model} loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("Enter Feature Values")

# Create input fields for all features
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Price", min_value=0.0, value=50.0, step=1.0)
    freight_value = st.number_input("Freight Value", min_value=0.0, value=15.0, step=0.1)
    payment_installments = st.number_input("Payment Installments", min_value=1.0, value=1.0, step=1.0)
    delivery_diff_than_estimated = st.number_input("Delivery Diff Than Estimated (days)", value=10.0, step=1.0)

with col2:
    reviewed_days = st.number_input("Reviewed Days", min_value=0.0, value=0.0, step=1.0)
    customer_state_enc = st.number_input("Customer State Encoded", min_value=0.0, max_value=1.0, value=0.78, step=0.01)
    product_category_name_enc = st.number_input("Product Category Encoded", min_value=0.0, value=7196.0, step=1.0)
    payment_type_enc = st.number_input("Payment Type Encoded", min_value=0.0, value=1.0, step=1.0)

# Predict button
if st.button("Predict", type="primary"):
    # Create input dataframe
    input_data = pd.DataFrame({
        'price': [price],
        'freight_value': [freight_value],
        'payment_installments': [payment_installments],
        'delivery_diff_than_estimated': [delivery_diff_than_estimated],
        'reviewed_days': [reviewed_days],
        'customer_state_enc': [customer_state_enc],
        'product_category_name_enc': [product_category_name_enc],
        'payment_type_enc': [payment_type_enc]
    })
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Try to get probability if available
        try:
            proba = model.predict_proba(input_data)[0]
            
            st.success(f"### Prediction: {prediction}")
            
            # Display probabilities
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Class': range(len(proba)),
                'Probability': proba
            })
            st.dataframe(prob_df, use_container_width=True)
            
            # Probability bar chart
            st.bar_chart(prob_df.set_index('Class'))
            
        except:
            # Model doesn't support probability
            st.success(f"### Prediction: {prediction}")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Show input summary
with st.expander("View Input Summary"):
    input_summary = pd.DataFrame({
        'Feature': ['price', 'freight_value', 'payment_installments', 
                   'delivery_diff_than_estimated', 'reviewed_days', 
                   'customer_state_enc', 'product_category_name_enc', 'payment_type_enc'],
        'Value': [price, freight_value, payment_installments, 
                 delivery_diff_than_estimated, reviewed_days, 
                 customer_state_enc, product_category_name_enc, payment_type_enc]
    })
    st.dataframe(input_summary, use_container_width=True)
