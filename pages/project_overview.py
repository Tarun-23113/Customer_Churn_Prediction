import streamlit as st
from PIL import Image

def app():
    st.header("Project Overview")
    st.markdown("This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners.The data is divided in multiple datasets for better understanding and organization. Please refer to the following data schema when working with it")
    try:
        img = Image.open("assets/schema.png")
        st.image(img, use_column_width=True)
    except Exception:
        st.warning("assets/schema.png not found")
    images = [
        "assets/customer_data.png",
        "assets/geolocation_data.png",
        "assets/order_data.png",
        "assets/order_items_data.png",
        "assets/order_payments_data.png",
        "assets/reviews_data.png",
        "assets/product_data.png",
        "assets/seller_id_data.png",
        "assets/product_category_data.png"
    ]
    idx = 0
    for r in range(3):
        cols = st.columns(3)
        for c in cols:
            try:
                c.image(Image.open(images[idx]), use_column_width=True)
            except Exception:
                c.write(images[idx] + " missing")
            idx += 1
    st.markdown("As we can see, the dataset contains numerous columns; however, based on the project requirements, we have selected only the most relevant features. The resulting dataset shown below represents the uncleaned merged version before any preprocessing or transformation.")
