import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Product Sales Dashboard", layout="wide")

# --- READ CSV DIRECTLY ---
df = pd.read_csv("data/amazon_products_sales_data_cleaned.csv")

# --- CLEAN / FEATURE ENGINEER ---
df["discount_percent"] = (1 - df["discounted_price"] / df["original_price"]) * 100
df["discount_percent"] = df["discount_percent"].clip(lower=0)
df["is_best_seller"] = df["is_best_seller"].astype(str)
df["is_sponsored"] = df["is_sponsored"].astype(str)
df["has_coupon"] = df["has_coupon"].astype(str)

# --- KPIs ---
avg_rating = df["product_rating"].mean()
total_reviews = df["total_reviews"].sum()
avg_discount = df["discount_percent"].mean()
best_seller_share = (df["is_best_seller"].str.lower() == "true").mean()

st.title("Amazon Product Sales Performance Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Rating", f"{avg_rating:.2f}")
c2.metric("Total Reviews", f"{total_reviews:,.0f}")
c3.metric("Avg Discount", f"{avg_discount:.1f}%")
c4.metric("Best Seller Share", f"{best_seller_share*100:.1f}%")

# --- CHARTS ---
st.subheader("Rating Distribution")
st.plotly_chart(px.histogram(df, x="product_rating", nbins=20), use_container_width=True)

st.subheader("Discount Distribution")
st.plotly_chart(px.histogram(df, x="discount_percent", nbins=20), use_container_width=True)

st.subheader("Units Purchased Last Month vs Rating")
st.plotly_chart(px.scatter(df, x="purchased_last_month", y="product_rating",
                           color="is_best_seller", hover_name="product_title"),
                use_container_width=True)

st.subheader("Average Discount by Coupon & Sponsorship")
st.plotly_chart(
    px.box(df, x="has_coupon", y="discount_percent", color="is_sponsored"),
    use_container_width=True
)
