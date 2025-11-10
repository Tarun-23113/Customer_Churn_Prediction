import streamlit as st

def app():
    st.header("📉 Churn Definition")

    st.markdown("""
    ### 🧠 What is Churn?
    Churn refers to customers who have **stopped purchasing or interacting**
    with the platform after their last order. Detecting churn helps in
    identifying at-risk customers for targeted retention strategies.
    """)

    st.markdown("""
    ### ⚙️ Churn Logic (Stepwise)
    1️⃣ **Identify** each customer's most recent completed order.
    2️⃣ **Set** a business-relevant inactivity threshold here we taken 90 days.
    3️⃣ **Label** customers with no new purchases within that period as *churned*.
    4️⃣ **Exclude** customers with returns, refunds, or unresolved issues from automatic labeling.
    """)

    st.markdown("""
    ### 💡 Why This Works
    This time-based definition is a **standard and practical approach**
    in e-commerce churn prediction — it directly measures disengagement
    and aligns well with business retention goals.
    """)

    st.info("Final churn labels (0 = Active, 1 = Churned) were derived using this standardized logic.")
