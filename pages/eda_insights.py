import streamlit as st
from PIL import Image

def app():
    st.title("EDA & Insights")
    st.markdown("Insights derived from EDA:")
    st.write("- Expensive or bulky items (furniture, home decor) tend to have higher churn")
    st.write("- Non-churners tend to review sooner after delivery")
    st.write("- Churn depends less on pricing and more on delivery/review dynamics")
    st.write("- Slightly higher order values could reduce churn")
    st.write("- If unusually high freight cost → higher churn risk")
    st.write("- Tree-based models work well due to mixed types and non-linear relationships")
    try:
        st.image(Image.open("assets_eda/eda_confusion_full.png"), use_column_width=True)
    except Exception:
        st.info("assets_eda/eda_confusion_full.png missing")
    cols = st.columns(3)
    imgs = ["assets_eda/categories_vs_churn.png","assets_eda/churn_imbalance.png","assets_eda/days_vs_churn_1.png"]
    for i,c in enumerate(cols):
        try:
            c.image(Image.open(imgs[i]), use_column_width=True)
        except Exception:
            c.write(imgs[i] + " missing")
    try:
        st.image(Image.open("assets_eda/product_distribution_full.png"), use_column_width=True)
    except Exception:
        st.info("assets_eda/product_distribution_full.png missing")
