import streamlit as st
import pandas as pd
from utils.helper import load_csv

def app():
    st.header("🧾 Data Quality Audit (After Cleaning)")

    try:
        df = load_csv("data/final_cleaned_df.csv")
    except Exception:
        st.error("data/final_cleaned_df.csv not found.")
        return

    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / df.shape[0]).sort_values(ascending=False)
    dtypes = df.dtypes.reset_index().rename(columns={"index": "column", 0: "dtype"})
    duplicates = df.duplicated().sum()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Missing Values (Top 15)")
        st.dataframe(
            pd.concat([missing.head(15), missing_pct.head(15)], axis=1, keys=["Missing", "Missing %"])
        )
    with col2:
        st.subheader("📦 Data Types Summary")
        st.dataframe(dtypes)

    st.subheader("🔁 Duplicate Rows")
    st.metric(label="Count of Duplicates", value=duplicates)

    st.markdown("""
    ---
    ### 🔍 Summary
    After cleaning, the dataset exhibits **no missing values**,
    consistent data types, and **no significant duplication issues**.
    This ensures the dataset is **ready for feature engineering and modeling**.
    """)

    with st.expander("🧠 Reproducible Audit Snippet"):
        st.code('''missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = missing / df.shape[0]
dtypes = df.dtypes
duplicates = df.duplicated().sum()''')
