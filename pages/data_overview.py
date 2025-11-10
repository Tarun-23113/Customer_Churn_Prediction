import streamlit as st
from utils.helper import load_csv

def app():
    try:
        df = load_csv("data/merged_df.csv")
    except Exception:
        st.error("data/merged_df.csv not found")
        return
    st.markdown(f"Total rows: {df.shape[0]}  |  Total columns: {df.shape[1]}")
    st.dataframe(df.head())
    st.markdown("### Final selected columns for modeling")
    st.write(list(df.columns))
