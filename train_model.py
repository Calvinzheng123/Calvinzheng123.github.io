# train_model.py — Reviews regressor with aligned metrics (log + linear)
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer

# -------------------- I/O --------------------
DATA = Path("data/amazon_products_sales_data_cleaned.csv")
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "reviews_model.pkl"
META_PATH  = MODEL_DIR / "reviews_meta.joblib"

# -------------------- Load -------------------
df = pd.read_csv(DATA)

# -------------------- Features ----------------
# core price/discount features
df["discount_percent"] = (1 - df["discounted_price"] / df["original_price"]) * 100
df["discount_percent"] = df["discount_percent"].clip(lower=0)
df["price_delta"] = (df["original_price"] - df["discounted_price"]).clip(lower=0)
df["price_ratio"] = np.where(df["original_price"] > 0,
                             df["discounted_price"] / df["original_price"], 1.0)

# booleans → 0/1
for c in ["has_coupon", "is_sponsored"]:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    else:
        df[c] = 0

need = [
    "product_rating", "purchased_last_month",
    "discounted_price", "original_price",
    "discount_percent", "price_delta", "price_ratio",
    "has_coupon", "is_sponsored", "product_category",
    "total_reviews"
]
missing = [c for c in need if c not in df.columns]
if missing:
    print("ERROR: missing columns:", missing)
    sys.exit(1)

# target (long-tailed) → log1p
y_lin = df["total_reviews"].clip(lower=0).fillna(0).astype(float)
y_log = np.log1p(y_lin)

# top-N category to cap OHE size
top_cats = df["product_category"].value_counts().nlargest(30).index
df["product_category_top"] = np.where(
    df["product_category"].isin(top_cats), df["product_category"], "Other"
)

num_cols = [
    "product_rating", "purchased_last_month",
    "discounted_price", "original_price",
    "discount_percent", "price_delta", "price_ratio",
    "has_coupon", "is_sponsored"
]
cat_cols = ["product_category_top"]

X = df[num_cols + cat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

pre = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ]
)

model = HistGradientBoostingRegressor(
    learning_rate=0.08,
    max_iter=400,
    min_samples_leaf=20,
    random_state=42
)

pipe = Pipeline([("pre", pre), ("model", model)])

# -------------------- Scorers -----------------
def r2_on_linear(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return r2_score(y_true, y_pred)

r2_linear_scorer = make_scorer(r2_on_linear, greater_is_better=True)

# -------------------- CV (log + linear) -------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2_log = cross_val_score(pipe, X, y_log, scoring="r2", cv=cv, n_jobs=-1)
cv_r2_lin = cross_val_score(pipe, X, y_log, scoring=r2_linear_scorer, cv=cv, n_jobs=-1)

print(f"CV R² (log, mean±std): {cv_r2_log.mean():.3f} ± {cv_r2_log.std():.3f}")
print(f"CV R² (lin, mean±std): {cv_r2_lin.mean():.3f} ± {cv_r2_lin.std():.3f}")

# -------------------- Holdout -----------------
Xtr, Xte, ytr_log, yte_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
pipe.fit(Xtr, ytr_log)

# log-space metrics (comparable to CV log)
pred_log = pipe.predict(Xte)
holdout_r2_log = r2_score(yte_log, pred_log)

# linear-space metrics (human-readable)
pred_lin = np.expm1(pred_log).clip(min=0)
yte_lin = np.expm1(yte_log)
holdout_mae_lin = mean_absolute_error(yte_lin, pred_lin)
holdout_r2_lin  = r2_score(yte_lin, pred_lin)

print(f"Holdout R² (log): {holdout_r2_log:.3f}")
print(f"Holdout MAE (lin): {holdout_mae_lin:,.2f}")
print(f"Holdout R²  (lin): {holdout_r2_lin:.3f}")

# -------------------- Save --------------------
joblib.dump(pipe, MODEL_PATH)
joblib.dump({
    "feat_cols": num_cols + cat_cols,
    "cat_col": "product_category_top",
    "target": "total_reviews",
    "target_transform": "log1p",
    "top_categories": list(top_cats)
}, META_PATH)

print(f"Saved -> {MODEL_PATH}")
print(f"Saved -> {META_PATH}")
