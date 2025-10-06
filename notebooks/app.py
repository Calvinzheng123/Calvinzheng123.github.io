import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Amazon Product Engagement", layout="wide")

# --- READ CSV DIRECTLY ---
# Just set your path here
df = pd.read_csv('/Users/calvi/Downloads/archive (4)/amazon_products_sales_data_cleaned.csv', parse_dates=["date"])

# --- BASIC CHECKS ---
df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

# --- KPIs ---
total_sales = df["sales"].sum()
avg_rating = df["rating"].mean()
num_reviews = df["reviews"].sum()

by_cat = df.groupby("category", as_index=False)["sales"].sum().sort_values("sales", ascending=False)
top_cat_share = by_cat["sales"].iloc[0] / total_sales if total_sales > 0 else 0

st.title("Amazon Product Engagement Dashboard")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"{total_sales:,.0f}")
c2.metric("Avg Rating", f"{avg_rating:.2f}")
c3.metric("Reviews", f"{num_reviews:,}")
c4.metric("Top Category Share", f"{top_cat_share:.0%}")

# --- CHARTS ---
st.subheader("Sales by Category")
st.plotly_chart(px.bar(by_cat, x="category", y="sales"), use_container_width=True)

st.subheader("Category Share")
st.plotly_chart(px.pie(by_cat, names="category", values="sales", hole=0.35), use_container_width=True)

st.subheader("Sales + Reviews Over Time")
by_month = df.groupby("month", as_index=False).agg({"sales": "sum", "reviews": "sum"})
st.plotly_chart(px.line(by_month, x="month", y=["sales", "reviews"], markers=True), use_container_width=True)

# --- TOP/BOTTOM PRODUCTS ---
fdf = df.copy()
fdf["engagement_score"] = (
    0.6 * (fdf["sales"] / (fdf["sales"].max() or 1))
    + 0.2 * (fdf["rating"] / 5.0)
    + 0.2 * (fdf["reviews"] / (fdf["reviews"].max() or 1))
)

prod_stats = (
    fdf.groupby(["product_name", "category", "brand"], as_index=False)
    .agg(
        sales=("sales", "sum"),
        rating=("rating", "mean"),
        reviews=("reviews", "sum"),
        engagement_score=("engagement_score", "mean"),
    )
    .sort_values("engagement_score", ascending=False)
)

left, right = st.columns(2)
left.markdown("**Top 15 by Engagement**")
left.dataframe(prod_stats.head(15), use_container_width=True)
right.markdown("**Bottom 15 by Engagement**")
right.dataframe(prod_stats.tail(15), use_container_width=True)

st.caption("Reads CSV directly from a fixed path. Adjust the path in pd.read_csv() if needed.")
