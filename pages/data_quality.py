import streamlit as st
import pandas as pd
from utils.helper import load_csv

def app():
    st.header("Data Quality Audit (Before Cleaning)")
    try:
        df = load_csv("data/merged_df.csv")
    except Exception:
        st.error("data/merged_df.csv not found")
        return
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / df.shape[0]).sort_values(ascending=False)
    dtypes = df.dtypes
    duplicates = df.duplicated().sum()
    st.subheader("Missing values (top 20)")
    st.dataframe(pd.concat([missing.head(20), missing_pct.head(20)], axis=1, keys=["missing","missing_pct"]))
    st.subheader("Data types")
    st.dataframe(dtypes.reset_index().rename(columns={"index":"column",0:"dtype"}))
    st.subheader("Duplicate rows")
    st.write(duplicates)
    st.markdown("Use these snippets locally to reproduce these audits if needed.")
    st.code('''missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = missing / df.shape[0]
dtypes = df.dtypes
duplicates = df.duplicated().sum()''')
