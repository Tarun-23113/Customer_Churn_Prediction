import streamlit as st
from utils.helper import load_image

def app():
    st.header("Data Cleaning Steps")

    st.markdown("""
    ### 🧹 Data Cleaning Overview
    After merging all datasets, several cleaning steps were performed to ensure high-quality input for modeling:
    """)

    st.image("assets_eda/data_overview.png", use_container_width=True)

    st.markdown("-------------------------------------")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("assets_eda/missing_values_1.png", caption="Missing Values in each columns", use_container_width=True)
    with col2:
        st.image("assets_eda/missing_values_2.png", caption="Missing Values in each columns", use_container_width=True)

    st.subheader("🔸 Handling Missing Values")
    st.markdown("""
    Handled through targeted imputation (for essential columns) or row/column removal (for sparse ones).
    """)

    col3, col4 = st.columns([1, 1])
    with col3:
        st.image("assets_eda/outliers_1.png", caption="Outliers", use_container_width=True)
    with col4:
        st.image("assets_eda/outliers_2.png", caption="Outliers", use_container_width=True)

    # --- Outlier Treatment Section ---
    st.subheader("🔸 Outlier Treatment")
    st.markdown("""
    Detected and capped for key numeric columns like delivery delay, freight value, and payment installments.
    """)

    # --- Redundant Columns Section ---
    st.subheader("🔸 Removing Redundant Columns")
    st.markdown("""
    Columns with very high cardinality or little predictive power were dropped 
    (e.g., product_description_length, product_photos_qty, payment_sequential).
    """)

    # --- Consistency Checks ---
    st.subheader("🔸 Consistency Checks")
    st.markdown("""
    Ensured consistent customer, order, and product mappings across datasets.  
    The cleaned dataset represents a balanced, noise-free version of the raw merged data — ready for feature engineering and model training.
    """)
