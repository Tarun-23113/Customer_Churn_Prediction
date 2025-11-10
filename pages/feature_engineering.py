import streamlit as st
from utils.helper import load_csv

def app():
    st.header("Encoding and Feature Engineering")
    st.markdown("Categorical encoding used (precomputed):")
    st.write("customer_state - target encoding")
    st.write("product_category_name - frequency encoding")
    st.write("payment_type - label encoding")

    st.markdown("Now we have this final data to train our model :")
    try:
        df = load_csv("data/data.csv")
        st.dataframe(df.head())
    except Exception:
        st.info("data.csv not provided")
