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

# ---------------------------------------------------------------------
# AMAZON BRAND / DARK THEME
# ---------------------------------------------------------------------
AMAZON_ORANGE = "#FF9900"
AMAZON_BLUE = "#146EB4"
AMAZON_DARK = "#131921"   # darker background
AMAZON_CARD = "#232F3E"   # card / block background
AMAZON_TEXT = "#F9FAFB"
AMAZON_MUTED = "#D1D5DB"

# Plotly: dark theme + Amazon-ish colors
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    AMAZON_ORANGE,
    AMAZON_BLUE,
    AMAZON_CARD,
    "#00A8E1",
    "#B12704",
]
px.defaults.color_continuous_scale = [
    "#2A2A2A",
    "#4B3B21",
    "#8A5A14",
    "#FF9900",
    "#F57C00",
    "#FFB74D",
]

# Global CSS to darken the whole app
st.markdown(
    f"""
    <style>
        /* Main background */
        section.main {{
            background-color: {AMAZON_DARK};
            color: {AMAZON_TEXT};
        }}

        body {{
            background-color: {AMAZON_DARK};
        }}

        /* Generic text */
        .stMarkdown, .stText, .stCaption, p, span {{
            color: {AMAZON_TEXT} !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {AMAZON_TEXT} !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {AMAZON_CARD};
            color: {AMAZON_TEXT};
        }}
        [data-testid="stSidebar"] * {{
            color: {AMAZON_TEXT} !important;
        }}

        /* Expander styling */
        details[data-testid="stExpander"] > summary {{
            background-color: {AMAZON_CARD};
            color: {AMAZON_TEXT};
            border-radius: 0.4rem;
        }}
        details[data-testid="stExpander"] > div {{
            background-color: {AMAZON_DARK};
            color: {AMAZON_TEXT};
        }}

        /* Metric cards */
        .stMetric > div {{
            border-radius: 10px;
            border: 1px solid #3B4252;
            padding: 0.5rem 0.75rem;
            background-color: {AMAZON_CARD};
        }}

        /* Horizontal rule */
        hr {{
            border-color: #374151;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    df["discount_percent"] = df["discount_percent"].clip(lower=0).fillna(0)
    df["price_delta"] = (df["original_price"] - df["discounted_price"]).clip(lower=0)
    df["price_ratio"] = np.where(
        df["original_price"] > 0,
        df["discounted_price"] / df["original_price"],
        1.0,
    )

    # normalize promo flags
    for c in ["is_best_seller", "is_sponsored", "has_coupon"]:
        if c in df.columns:
            norm = df[c].astype(str).str.strip().str.lower()
            df[c] = norm
            df[f"{c}_flag"] = norm.isin(["true", "1", "yes", "y", "coupon", "has coupon"])
        else:
            df[f"{c}_flag"] = False

    # human-readable labels (avoid "undefined")
    df["sponsored_label"] = np.where(
        df["is_sponsored_flag"],
        "Sponsored ad",
        "Organic result",
    )
    df["coupon_label"] = np.where(
        df["has_coupon_flag"],
        "Has coupon",
        "No coupon",
    )

    # cap category list so UI stays readable
    top = df["product_category"].value_counts().nlargest(30).index
    df["_cat_top"] = np.where(
        df["product_category"].isin(top),
        df["product_category"],
        "Other",
    )
    return df, list(top)


@st.cache_resource
def load_model():
    """Try to load model. If it fails, try to retrain. If that fails, return (None, None)."""
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            meta = joblib.load(META_PATH)
            return m, meta
        except Exception as e:
            st.warning(
                f"Found model files but couldn't load them here ({type(e).__name__}). "
                "Will try retrain."
            )

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
st.sidebar.markdown(
    "### 🛍️ Amazon Engagement\n"
    "<span style='font-size:0.9rem;'>Filter down to categories, price bands, and promo setups.</span>",
    unsafe_allow_html=True,
)

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
    (p5, p95),
)

coupon = st.sidebar.selectbox("Has Coupon", ["Any", "Yes", "No"])
sponsored = st.sidebar.selectbox("Sponsored", ["Any", "Yes", "No"])


def flag_mask(series_flag: pd.Series, choice: str) -> pd.Series:
    if choice == "Any":
        return pd.Series(True, index=series_flag.index)
    return series_flag if choice == "Yes" else ~series_flag


mask = (
    df["_cat_top"].isin(cats if cats else top_cats + ["Other"])
    & df["product_rating"].between(rmin, rmax)
    & df["original_price"].between(pmin, pmax)
)
mask &= flag_mask(df["has_coupon_flag"], coupon)
mask &= flag_mask(df["is_sponsored_flag"], sponsored)

dfv = df.loc[mask].copy()

# ---------------------------------------------------------------------
# HEADER / STORY
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div style="background-color:{AMAZON_CARD}; padding: 1.1rem 1.4rem; border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid #3B4252;">
        <h1 style="color:{AMAZON_TEXT}; margin:0;">🛒 Amazon Product Engagement</h1>
        <p style="color:{AMAZON_MUTED}; margin:0.4rem 0 0;">
            See how <b>ratings</b>, <b>discounts</b>, <b>coupons</b>, and <b>sponsorships</b> line up with
            <b>customer engagement</b> (total reviews) for the slice you care about.
        </p>
        <p style="color:{AMAZON_MUTED}; margin:0.2rem 0 0; font-size:0.9rem;">
            Audience: Category & merchandising managers making decisions on promos, coupons, and paid placements.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How to read this dashboard (story flow)", expanded=True):
    st.markdown(
        """
        1. **Start with the KPIs** to understand this slice’s overall health.  
        2. **Check the rating distribution** to see whether this slice is trustworthy or noisy.  
        3. **Use the purchases vs rating view** to see how sponsorship and coupons show up.  
        4. **Run what-if scenarios** with the prediction tool for pricing and promo choices.  
        5. **Use the model performance section** if you care about how reliable the predictions are.
        """
    )

# ---------------------------------------------------------------------
# KPI ROW
# ---------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Rating", f"{dfv['product_rating'].mean():.2f}")
c2.metric("Total Reviews", f"{dfv['total_reviews'].sum():,.0f}")
c3.metric("Avg Discount", f"{dfv['discount_percent'].mean():.1f}%")
best_share = dfv["is_best_seller_flag"].mean()
c4.metric("Best-Seller Share", f"{best_share * 100:.1f}%")

st.caption(
    "High Avg Rating with low Total Reviews usually means a **visibility problem** more than a **satisfaction problem**."
)

# ---------------------------------------------------------------------
# MAIN VISUALS
# ---------------------------------------------------------------------
st.markdown("### 1️⃣ Rating distribution — how trustworthy is this slice?")

fig_rating = px.histogram(dfv, x="product_rating", nbins=20)
fig_rating.update_layout(
    xaxis_title="Rating",
    yaxis_title="Number of products",
    margin=dict(l=20, r=20, t=40, b=40),
)
st.plotly_chart(fig_rating, use_container_width=True)

st.markdown(
    "This tells you whether this slice is stacked with 4.5★+ products or has a lot of mediocre items mixed in."
)

st.markdown("### 2️⃣ Purchases vs rating — how do sponsored items behave?")

# WEBGL FIX: SVG
fig_scatter = px.scatter(
    dfv,
    x="purchased_last_month",
    y="product_rating",
    color="sponsored_label",  # clean labels
    hover_data={
        "product_title": True,
        "total_reviews": True,
        "discount_percent": ":.1f",
        "coupon_label": True,
    },
    render_mode="svg",
    labels={"sponsored_label": "Placement"},
)
fig_scatter.update_layout(
    xaxis_title="Purchased Last Month",
    yaxis_title="Rating",
    margin=dict(l=20, r=20, t=40, b=40),
    legend_title_text="Placement",
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    """
    - **Top-right:** highly rated, high-purchase products — protect these with inventory and strong positioning.  
    - **High rating, low purchases:** good candidates for **coupons or sponsorship**.  
    - Compare **Sponsored ad** vs **Organic result** clusters to see if ads are pushing low-quality items.
    """
)

st.divider()

# ---------------------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------------------
st.markdown("### 🔮 3️⃣ What-if: predict total reviews for a product setup")

if model is None or meta is None:
    st.warning(
        "Model not available in this environment. Run `python train_model.py` locally and refresh."
    )
else:
    feat_cols = meta["feat_cols"]
    cat_col = meta.get("cat_col", "product_category_top")
    tops = meta.get("top_categories", [])

    col_a, col_b, col_c = st.columns(3)
    rating_in = col_a.number_input("Rating", 0.0, 5.0, 4.2, 0.1)
    purchased_in = col_b.number_input("Purchased Last Month", 0, 200000, 30, 5)
    original_in = col_c.number_input(
        "Original Price", 0.0, 10000.0, 59.99, 1.0, format="%.2f"
    )

    col_d, col_e, col_f = st.columns(3)
    discounted_in = col_d.number_input(
        "Discounted Price", 0.0, 10000.0, 39.99, 1.0, format="%.2f"
    )
    has_cpn_in = col_e.checkbox("Has Coupon", True)
    is_spon_in = col_f.checkbox("Sponsored Placement", False)

    category_in = st.selectbox(
        "Category", options=(tops + ["Other"]) if tops else ["Other"]
    )

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
        X = (
            pd.DataFrame([row])[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )
        pred = np.expm1(model.predict(X)[0]).clip(min=0)
        st.metric("Predicted Reviews", f"{pred:,.0f}")

st.markdown(
    "_Use this to sanity-check ideas like: ‘If I sponsor this 4.5★ product and add a coupon, "
    "am I likely to see a meaningful lift in reviews over time?’_"
)

# ---------------------------------------------------------------------
# MODEL DIAGNOSTICS
# ---------------------------------------------------------------------
st.markdown("### 🧪 4️⃣ Model performance — how reliable is that prediction?")

with st.expander("Actual vs Predicted (log–log)", expanded=False):
    if model is None or meta is None:
        st.info("Model not loaded — skipping diagnostics.")
    else:
        s = df.sample(min(4000, len(df)), random_state=42).copy()

        if "has_coupon_flag" in s.columns:
            s["has_coupon"] = s["has_coupon_flag"].astype(int)
        else:
            s["has_coupon"] = binarize(
                s.get("has_coupon", pd.Series(0, index=s.index))
            )

        if "is_sponsored_flag" in s.columns:
            s["is_sponsored"] = s["is_sponsored_flag"].astype(int)
        else:
            s["is_sponsored"] = binarize(
                s.get("is_sponsored", pd.Series(0, index=s.index))
            )

        s["discount_percent"] = (
            (1 - s["discounted_price"] / s["original_price"]) * 100
        ).clip(lower=0)
        s["price_delta"] = (
            s["original_price"] - s["discounted_price"]
        ).clip(lower=0)
        s["price_ratio"] = np.where(
            s["original_price"] > 0,
            s["discounted_price"] / s["original_price"],
            1.0,
        )

        cat_col = meta.get("cat_col", "product_category_top")
        tops = meta.get("top_categories", [])
        s[cat_col] = np.where(
            s["product_category"].isin(tops), s["product_category"], "Other"
        )

        feat_cols = meta["feat_cols"]
        need = set(feat_cols) | {"total_reviews"}
        if any(c not in s.columns for c in need):
            st.info("Missing columns for diagnostics; skipped.")
        else:
            Xs = (
                s[list(feat_cols)]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
            y_true = s["total_reviews"].astype(float).to_numpy()
            y_pred = np.expm1(model.predict(Xs)).astype(float)

            finite = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true, y_pred = y_true[finite], y_pred[finite]

            if len(y_true) < 2:
                st.info("Not enough valid rows to compute diagnostics.")
            else:
                from sklearn.metrics import r2_score, mean_absolute_error

                r2 = r2_score(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                st.write(
                    f"R²: **{r2:.3f}** · "
                    f"MAE: **{mae:,.0f}** reviews"
                )

                eps = 1.0
                diag_df = pd.DataFrame(
                    {
                        "log_actual": np.log10(y_true + eps),
                        "log_pred": np.log10(y_pred + eps),
                    }
                )

                # SVG again for WebGL issues
                fig = px.scatter(
                    diag_df,
                    x="log_actual",
                    y="log_pred",
                    labels={
                        "log_actual": "log10(actual + 1)",
                        "log_pred": "log10(predicted + 1)",
                    },
                    title="Actual vs Predicted Reviews (log–log)",
                    render_mode="svg",
                )
                lo = float(
                    np.nanmin(
                        [diag_df["log_actual"].min(), diag_df["log_pred"].min()]
                    )
                )
                hi = float(
                    np.nanmax(
                        [diag_df["log_actual"].max(), diag_df["log_pred"].max()]
                    )
                )
                line = np.linspace(lo, hi, 200)
                fig.add_trace(
                    go.Scatter(
                        x=line,
                        y=line,
                        mode="lines",
                        name="y = x",
                        line=dict(dash="dash", color=AMAZON_ORANGE),
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# DESIGN NOTES / AI USE
# ---------------------------------------------------------------------
with st.expander("Design notes & AI transparency (for grading)", expanded=False):
    st.markdown(
        """
        **Audience & Intent**

        - Audience: Amazon category & merchandising managers deciding where to invest in promos and placements.
        - Intent: Separate *visibility* problems from *satisfaction* problems and give a simple what-if tool
          to estimate review volume given rating, pricing, and promo settings.

        **Design Choices**

        - Dark Amazon-inspired theme (navy background, orange accents) for a cohesive, branded look.
        - Single-column flow: KPIs → distribution → relationship → what-if → diagnostics.
        - Limited visuals with clear labels, and richer details pushed into hover tooltips and expanders.

        **Reflection**

        - Strong fit for decision-making: every chart answers a specific question.
        - Hardest part: balancing interactivity with keeping the story tight.
        - Next step (if I had more time): add time trends and product lifecycle segmentation (new vs long-tail).
        """
    )

st.markdown("---")
with st.expander("Notes on tools / AI use"):
    st.markdown(
        """
        I used **ChatGPT** to help with Streamlit / Plotly theming and wording for the design sections.
        The metric definitions, feature engineering, and modeling decisions are mine; AI was used as a support tool.
        """
    )

st.page_link(
    "https://calvinzheng123.github.io/",
    label="← Back to portfolio",
    icon="🏠",
)
