import streamlit as st

def app():
    st.header("🧹 Data Cleaning Steps")
    st.markdown("""
    After merging all datasets, several cleaning steps were performed to ensure high-quality input for modeling.
    """)

    st.image("assets_eda/data_overview.png", caption="Overview of merged dataset")
    st.markdown("---")

    st.subheader("🔸 Handling Missing Values")
    col1, col2 = st.columns(2)
    with col1:
        st.image("assets_eda/missing_values_1.png", caption="Missing Values (Visualization 1)")
    with col2:
        st.image("assets_eda/missing_value_2.png", caption="Missing Values (Visualization 2)")

    st.markdown("""
    Missing data was addressed through a combination of:
    - **Targeted imputation** for critical features (e.g., freight value, review scores)
    - **Row/column removal** for sparse or redundant entries
    """)

    st.subheader("🔸 Outlier Treatment")
    col3, col4 = st.columns(2)
    with col3:
        st.image("assets_eda/outliers_1.png", caption="Outliers in Key Numeric Columns")
    with col4:
        st.image("assets_eda/outliers_2.png", caption="Outlier Detection Results")

    st.markdown("""
    Outliers were capped or treated for columns such as:
    - **Delivery Delay**
    - **Freight Value**
    - **Payment Installments**
    """)

    st.subheader("🔸 Removing Redundant Columns")
    st.markdown("""
    Columns with high cardinality or little predictive value were dropped:
    - `product_description_length`
    - `product_photos_qty`
    - `payment_sequential`
    """)

    st.subheader("🔸 Consistency Checks")
    st.markdown("""
    Ensured consistent relationships between customers, orders, and products across datasets.  
    The final dataset is **clean, reliable, and ready for feature engineering and modeling.**
    """)