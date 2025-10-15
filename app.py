# app.py — Amazon Dashboard
import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
import joblib

st.set_page_config(page_title="Amazon Product Dashboard", layout="wide")

DATA = Path("data/amazon_products_sales_data_cleaned.csv")

from pathlib import Path
import subprocess, sys, joblib

MODEL_PATH = Path("model/reviews_model.pkl")
META_PATH  = Path("model/reviews_meta.joblib")

@st.cache_resource
def load_model():
    # 1) Try load
    try:
        if MODEL_PATH.exists() and META_PATH.exists():
            m = joblib.load(MODEL_PATH)
            meta = joblib.load(META_PATH)
            return m, meta
    except Exception as e:
        st.warning(f"Model load failed in this environment. Re-training here. Details: {type(e).__name__}")

    # 2) Retrain in *this* Python/env
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "train_model.py"])

    # 3) Load the freshly trained model
    m = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return m, meta

model, meta = load_model()

# Functions to help with loading and processing data
def binarize(s: pd.Series) -> pd.Series:
    m = {"true":1,"1":1,"yes":1,"y":1,"coupon":1,"has coupon":1,
         "false":0,"0":0,"no":0,"n":0,"no coupon":0}
    return s.astype(str).str.strip().str.lower().map(m).fillna(0).astype(int)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df["discount_percent"] = (1 - df["discounted_price"] / df["original_price"]) * 100
    df["discount_percent"] = df["discount_percent"].clip(lower=0)
    df["price_delta"] = (df["original_price"] - df["discounted_price"]).clip(lower=0)
    df["price_ratio"] = np.where(df["original_price"] > 0, df["discounted_price"]/df["original_price"], 1.0)
    for c in ["is_best_seller","is_sponsored","has_coupon"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    top = df["product_category"].value_counts().nlargest(30).index
    df["_cat_top"] = np.where(df["product_category"].isin(top), df["product_category"], "Other")
    return df, list(top)

df, top_cats = load_data()

#Sidebar Filters
st.sidebar.header("Filters")
cats = st.sidebar.multiselect("Category", options=top_cats+["Other"],
                              default=top_cats[:8] if len(top_cats)>8 else top_cats+["Other"])
rmin, rmax = st.sidebar.slider("Rating", 0.0, 5.0, (3.5, 5.0), 0.1)
p5, p95 = float(np.nanpercentile(df["original_price"],5)), float(np.nanpercentile(df["original_price"],95))
pmin, pmax = st.sidebar.slider("Original Price", float(df["original_price"].min()),
                               float(df["original_price"].max()), (p5, p95))
coupon = st.sidebar.selectbox("Has Coupon", ["Any","Yes","No"])
sponsored = st.sidebar.selectbox("Sponsored", ["Any","Yes","No"])

def flag_mask(series, choice):
    if choice=="Any": return pd.Series(True, index=series.index)
    yes = series.astype(str).str.lower().isin(["true","1","yes","y","coupon","has coupon"])
    return yes if choice=="Yes" else ~yes

mask = (
    df["_cat_top"].isin(cats if cats else top_cats+["Other"]) &
    df["product_rating"].between(rmin, rmax) &
    df["original_price"].between(pmin, pmax)
)
if "has_coupon" in df.columns:   mask &= flag_mask(df["has_coupon"], coupon)
if "is_sponsored" in df.columns: mask &= flag_mask(df["is_sponsored"], sponsored)

dfv = df.loc[mask].copy()

#KPI's and metrics
st.title("Amazon Product Engagement")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Avg Rating", f"{dfv['product_rating'].mean():.2f}")
c2.metric("Total Reviews", f"{dfv['total_reviews'].sum():,.0f}")
c3.metric("Avg Discount", f"{dfv['discount_percent'].mean():.1f}%")
best_share = (dfv["is_best_seller"].str.lower()=="true").mean() if "is_best_seller" in dfv.columns else 0.0
c4.metric("Best-Seller Share", f"{best_share*100:.1f}%")

with st.expander("About"):
    st.write("Audience: merchandising/category managers. Purpose: track engagement and test how pricing/discount/coupons/sponsorship relate to reviews.")

# Charts
st.subheader("Rating Distribution")
st.plotly_chart(px.histogram(dfv, x="product_rating", nbins=20)
                .update_layout(xaxis_title="Rating", yaxis_title="Count"),
                use_container_width=True)

st.subheader("Purchases vs Rating")
st.plotly_chart(px.scatter(dfv, x="purchased_last_month", y="product_rating",
                           color=("is_sponsored" if "is_sponsored" in dfv.columns else None),
                           hover_data={"product_title":True,"total_reviews":True,"discount_percent":":.1f"})
                .update_layout(xaxis_title="Purchased Last Month", yaxis_title="Rating"),
                use_container_width=True)

st.divider()

# getting model from model.py
@st.cache_resource
def load_model():
    if not (MODEL_PATH.exists() and META_PATH.exists()): return None, None
    return joblib.load(MODEL_PATH), joblib.load(META_PATH)

model, meta = load_model()

st.header("🔮 Predict Total Reviews")
if model is None or meta is None:
    st.warning("Train the model first: `python train_model.py` (saves model/).")
else:
    feat_cols = meta["feat_cols"]
    cat_col = meta.get("cat_col","product_category_top")
    tops = meta.get("top_categories", [])

    # predictor UI
    a,b,c = st.columns(3)
    rating = a.number_input("Rating", 0.0, 5.0, 4.2, 0.1)
    purchased = b.number_input("Purchased Last Month", 0, 200000, 30, 5)
    original = c.number_input("Original Price", 0.0, 10000.0, 59.99, 1.0, format="%.2f")

    d,e,f = st.columns(3)
    discounted = d.number_input("Discounted Price", 0.0, 10000.0, 39.99, 1.0, format="%.2f")
    has_cpn = e.checkbox("Has Coupon", True)
    is_spon = f.checkbox("Sponsored", False)

    category = st.selectbox("Category", options=(tops+["Other"]) if tops else ["Other"])

    disc_pct = (1 - (discounted/original))*100 if original>0 else 0.0
    st.caption(f"Discount %: **{max(disc_pct,0):.1f}%**")

    if st.button("Predict"):
        row = {
            "product_rating": rating,
            "purchased_last_month": purchased,
            "discounted_price": discounted,
            "original_price": original,
            "discount_percent": max(disc_pct,0),
            "price_delta": max(original-discounted,0.0),
            "price_ratio": (discounted/original) if original>0 else 1.0,
            "has_coupon": int(has_cpn),
            "is_sponsored": int(is_spon),
            cat_col: category if category in tops else "Other",
        }
        X = pd.DataFrame([row])[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0)
        pred = np.expm1(model.predict(X)[0])
        st.metric("Predicted Reviews", f"{pred:,.0f}")

#comparing model to actual
st.subheader("Model Performance")
with st.expander("Actual vs Predicted (log–log)"):
    s = df.sample(min(4000, len(df)), random_state=42).copy()

    s["has_coupon"]   = binarize(s.get("has_coupon",   pd.Series(0, index=s.index)))
    s["is_sponsored"] = binarize(s.get("is_sponsored", pd.Series(0, index=s.index)))

    # engineered features
    s["discount_percent"] = (1 - s["discounted_price"]/s["original_price"]) * 100
    s["discount_percent"] = s["discount_percent"].clip(lower=0)
    s["price_delta"] = (s["original_price"] - s["discounted_price"]).clip(lower=0)
    s["price_ratio"] = np.where(s["original_price"] > 0,
                                s["discounted_price"]/s["original_price"], 1.0)
    s[cat_col] = np.where(s["product_category"].isin(tops), s["product_category"], "Other")

    need = set(feat_cols) | {"total_reviews"}
    if any(c not in s.columns for c in need):
        st.info("Missing columns for diagnostics; skip.")
    else:
        Xs = s[list(feat_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_true = s["total_reviews"].astype(float).to_numpy()
        y_pred = np.expm1(model.predict(Xs)).astype(float)

        # filter to finite values only
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[finite], y_pred[finite]

        if len(y_true) < 2:
            st.info("Not enough valid rows to compute diagnostics after filtering.")
        else:
            from sklearn.metrics import r2_score, mean_absolute_error
            st.write(f"R²: **{r2_score(y_true, y_pred):.3f}** · MAE: **{mean_absolute_error(y_true, y_pred):,.0f}**")

            # log–log scatter with identity line
            eps = 1.0
            x = np.log10(y_true + eps)
            y = np.log10(y_pred + eps)
            diag_df = pd.DataFrame({"log_actual": x, "log_pred": y})

            fig = px.scatter(
                diag_df,
                x="log_actual",
                y="log_pred",
                labels={"log_actual": "log10(actual+1)", "log_pred": "log10(pred+1)"},
                title="Actual vs Predicted (log–log)"
            )
            lim_lo = float(np.nanmin([diag_df["log_actual"].min(), diag_df["log_pred"].min()]))
            lim_hi = float(np.nanmax([diag_df["log_actual"].max(), diag_df["log_pred"].max()]))
            line = np.linspace(lim_lo, lim_hi, 200)
            fig.add_trace(go.Scatter(x=line, y=line, mode="lines", name="y = x", line=dict(dash="dash")))
            st.plotly_chart(fig, use_container_width=True)

