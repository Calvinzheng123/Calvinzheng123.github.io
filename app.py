# app.py — Amazon Product Engagement Dashboard
# Calvin Zheng
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys, subprocess, joblib

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Amazon Product Dashboard",
    layout="wide",
    page_icon="🛒",
)

# consistent chart look
px.defaults.template = "plotly_white"
px.defaults.color_continuous_scale = "Blues"

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DATA = Path("data/amazon_products_sales_data_cleaned.csv")
MODEL_PATH = Path("model/reviews_model.pkl")
META_PATH = Path("model/reviews_meta.joblib")


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def binarize(s: pd.Series) -> pd.Series:
    m = {
        "true": 1, "1": 1, "yes": 1, "y": 1, "coupon": 1, "has coupon": 1,
        "false": 0, "0": 0, "no": 0, "n": 0, "no coupon": 0
    }
    return s.astype(str).str.strip().str.lower().map(m).fillna(0).astype(int)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    # feature engineering
    df["discount_percent"] = (1 - df["discounted_price"] / df["original_price"]) * 100
    df["discount_percent"] = df["discount_percent"].clip(lower=0)
    df["price_delta"] = (df["original_price"] - df["discounted_price"]).clip(lower=0)
    df["price_ratio"] = np.where(df["original_price"] > 0,
                                 df["discounted_price"] / df["original_price"],
                                 1.0)
    for c in ["is_best_seller", "is_sponsored", "has_coupon"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # cap category list so UI stays readable
    top = df["product_category"].value_counts().nlargest(30).index
    df["_cat_top"] = np.where(df["product_category"].isin(top),
                              df["product_category"], "Other")
    return df, list(top)


@st.cache_resource
def load_model():
    """Try to load model. If it fails, try to retrain. If that fails, return (None, None)."""
    # 1) try to load ready-made model
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            meta = joblib.load(META_PATH)
            return m, meta
        except Exception as e:
            st.warning(f"Found model files but couldn't load them here ({type(e).__name__}). Will try retrain.")

    # 2) try retrain in current python (may fail on local base env)
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([sys.executable, "train_model.py"])
        m = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return m, meta
    except Exception as e:
        st.error(
            "Model isn’t available in this Python environment and auto-training failed.\n"
            "Run `python train_model.py` in the env that has scikit-learn, then refresh."
        )
        st.caption(f"(debug: {type(e).__name__})")
        return None, None


# ---------------------------------------------------------------------
# LOAD DATA / MODEL
# ---------------------------------------------------------------------
df, top_cats = load_data()
model, meta = load_model()

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.header("Filters")

cats = st.sidebar.multiselect(
    "Category",
    options=top_cats + ["Other"],
    default=top_cats[:8] if len(top_cats) > 8 else top_cats + ["Other"],
)

rmin, rmax = st.sidebar.slider("Rating", 0.0, 5.0, (3.5, 5.0), 0.1)

p5 = float(np.nanpercentile(df["original_price"], 5))
p95 = float(np.nanpercentile(df["original_price"], 95))
pmin, pmax = st.sidebar.slider(
    "Original Price",
    float(df["original_price"].min()),
    float(df["original_price"].max()),
    (p5, p95)
)

coupon = st.sidebar.selectbox("Has Coupon", ["Any", "Yes", "No"])
sponsored = st.sidebar.selectbox("Sponsored", ["Any", "Yes", "No"])


def flag_mask(series, choice):
    if choice == "Any":
        return pd.Series(True, index=series.index)
    yes = series.astype(str).str.lower().isin(["true", "1", "yes", "y", "coupon", "has coupon"])
    return yes if choice == "Yes" else ~yes


mask = (
    df["_cat_top"].isin(cats if cats else top_cats + ["Other"]) &
    df["product_rating"].between(rmin, rmax) &
    df["original_price"].between(pmin, pmax)
)
if "has_coupon" in df.columns:
    mask &= flag_mask(df["has_coupon"], coupon)
if "is_sponsored" in df.columns:
    mask &= flag_mask(df["is_sponsored"], sponsored)

dfv = df.loc[mask].copy()

# ---------------------------------------------------------------------
# HEADER / STORY
# ---------------------------------------------------------------------
st.title("🛒 Amazon Product Engagement")

st.markdown(
    """
    **Audience:** merchandising / category managers (or anyone doing marketplace analytics).  
    **Goal:** see how **ratings, discounts, coupons, and sponsorships** line up with **engagement** (reviews).
    Filters on the left let you look at specific slices (categories, price bands, promos).
    """
)

# ---------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Rating", f"{dfv['product_rating'].mean():.2f}")
c2.metric("Total Reviews", f"{dfv['total_reviews'].sum():,.0f}")
c3.metric("Avg Discount", f"{dfv['discount_percent'].mean():.1f}%")
best_share = (
    (dfv["is_best_seller"].str.lower() == "true").mean()
    if "is_best_seller" in dfv.columns else 0.0
)
c4.metric("Best-Seller Share", f"{best_share * 100:.1f}%")

st.caption(
    "If Avg Rating is high but Total Reviews is low for your slice, it usually means **visibility**, not **satisfaction**, is the issue."
)

# ---------------------------------------------------------------------
# MAIN VISUALS
# ---------------------------------------------------------------------
st.markdown("### 1️⃣ Rating distribution — ratings are compressed near 4–5")

fig_rating = (
    px.histogram(dfv, x="product_rating", nbins=20)
    .update_layout(xaxis_title="Rating", yaxis_title="Count", margin=dict(l=20, r=20, t=40, b=40))
)
st.plotly_chart(fig_rating, use_container_width=True)

st.markdown("### 2️⃣ Purchases vs rating — colored by sponsorship to see if promoted items behave differently")

fig_scatter = (
    px.scatter(
        dfv,
        x="purchased_last_month",
        y="product_rating",
        color=("is_sponsored" if "is_sponsored" in dfv.columns else None),
        hover_data={
            "product_title": True,
            "total_reviews": True,
            "discount_percent": ":.1f"
        },
    )
    .update_layout(
        xaxis_title="Purchased Last Month",
        yaxis_title="Rating",
        margin=dict(l=20, r=20, t=40, b=40)
    )
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------------------
st.markdown("### 🔮 3️⃣ Predict total reviews (what-if)")

if model is None or meta is None:
    st.warning("Model not available in this environment. Train locally (`python train_model.py`) and refresh.")
else:
    feat_cols = meta["feat_cols"]
    cat_col = meta.get("cat_col", "product_category_top")
    tops = meta.get("top_categories", [])

    col_a, col_b, col_c = st.columns(3)
    rating_in = col_a.number_input("Rating", 0.0, 5.0, 4.2, 0.1)
    purchased_in = col_b.number_input("Purchased Last Month", 0, 200000, 30, 5)
    original_in = col_c.number_input("Original Price", 0.0, 10000.0, 59.99, 1.0, format="%.2f")

    col_d, col_e, col_f = st.columns(3)
    discounted_in = col_d.number_input("Discounted Price", 0.0, 10000.0, 39.99, 1.0, format="%.2f")
    has_cpn_in = col_e.checkbox("Has Coupon", True)
    is_spon_in = col_f.checkbox("Sponsored", False)

    category_in = st.selectbox("Category", options=(tops + ["Other"]) if tops else ["Other"])

    disc_pct = (1 - (discounted_in / original_in)) * 100 if original_in > 0 else 0.0
    st.caption(f"Discount %: **{max(disc_pct, 0):.1f}%**")

    if st.button("Predict"):
        row = {
            "product_rating": rating_in,
            "purchased_last_month": purchased_in,
            "discounted_price": discounted_in,
            "original_price": original_in,
            "discount_percent": max(disc_pct, 0),
            "price_delta": max(original_in - discounted_in, 0.0),
            "price_ratio": (discounted_in / original_in) if original_in > 0 else 1.0,
            "has_coupon": int(has_cpn_in),
            "is_sponsored": int(is_spon_in),
            cat_col: category_in if category_in in tops else "Other",
        }
        X = pd.DataFrame([row])[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        pred = np.expm1(model.predict(X)[0]).clip(min=0)
        st.metric("Predicted Reviews", f"{pred:,.0f}")

# ---------------------------------------------------------------------
# MODEL DIAGNOSTICS
# ---------------------------------------------------------------------
st.markdown("### 🧪 4️⃣ Model performance")

with st.expander("Actual vs Predicted (log–log)"):
    if model is None or meta is None:
        st.info("Model not loaded — skipping diagnostics.")
    else:
        s = df.sample(min(4000, len(df)), random_state=42).copy()

        s["has_coupon"] = binarize(s.get("has_coupon", pd.Series(0, index=s.index)))
        s["is_sponsored"] = binarize(s.get("is_sponsored", pd.Series(0, index=s.index)))
        s["discount_percent"] = (1 - s["discounted_price"] / s["original_price"]) * 100
        s["discount_percent"] = s["discount_percent"].clip(lower=0)
        s["price_delta"] = (s["original_price"] - s["discounted_price"]).clip(lower=0)
        s["price_ratio"] = np.where(s["original_price"] > 0,
                                    s["discounted_price"] / s["original_price"], 1.0)

        cat_col = meta.get("cat_col", "product_category_top")
        tops = meta.get("top_categories", [])
        s[cat_col] = np.where(s["product_category"].isin(tops), s["product_category"], "Other")

        feat_cols = meta["feat_cols"]
        need = set(feat_cols) | {"total_reviews"}
        if any(c not in s.columns for c in need):
            st.info("Missing columns for diagnostics; skipped.")
        else:
            Xs = s[list(feat_cols)].replace([np.inf, -np.inf], np.nan).fillna(0)
            y_true = s["total_reviews"].astype(float).to_numpy()
            y_pred = np.expm1(model.predict(Xs)).astype(float)

            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true, y_pred = y_true[finite], y_pred[finite]

            if len(y_true) < 2:
                st.info("Not enough valid rows to compute diagnostics.")
            else:
                from sklearn.metrics import r2_score, mean_absolute_error
                st.write(
                    f"R²: **{r2_score(y_true, y_pred):.3f}** · "
                    f"MAE: **{mean_absolute_error(y_true, y_pred):,.0f}**"
                )

                eps = 1.0
                diag_df = pd.DataFrame({
                    "log_actual": np.log10(y_true + eps),
                    "log_pred": np.log10(y_pred + eps)
                })
                fig = px.scatter(
                    diag_df,
                    x="log_actual",
                    y="log_pred",
                    labels={"log_actual": "log10(actual+1)", "log_pred": "log10(pred+1)"},
                    title="Actual vs Predicted (log–log)",
                )
                lo = float(np.nanmin([diag_df["log_actual"].min(), diag_df["log_pred"].min()]))
                hi = float(np.nanmax([diag_df["log_actual"].max(), diag_df["log_pred"].max()]))
                line = np.linspace(lo, hi, 200)
                fig.add_trace(
                    go.Scatter(x=line, y=line, mode="lines", name="y = x", line=dict(dash="dash"))
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# DISCLOSURE / PORTFOLIO
# ---------------------------------------------------------------------
with st.expander("Notes on tools / AI use"):
    st.markdown(
        """
        I used **ChatGPT** to help debug Streamlit deployment (model not loading on Cloud / version mismatches)
        and to tighten up the wording for the design/report parts.  
        The actual dashboard logic (filters, KPIs, feature engineering, and the idea to predict reviews from
        price + promo signals) was done by me. AI was a support tool, not the source of the analysis.
        """
    )

st.markdown("---")
st.page_link("https://calvin-zheng.netlify.app/", label="← Back to portfolio", icon="🏠")
