# adoption_model.py
# ==========================================================
# Predict product adoption with Logistic Regression
# Steps:
#   1) Load sample data from Supabase
#   2) Build a client-level dataset (features + target)
#   3) Split into train/test; train logistic regression
#   4) Evaluate (accuracy, precision, recall, ROC AUC, pseudo-R²)
#   5) Explain key drivers via coefficients
#   6) Segment clients: High / Medium / Low likelihood
#
# Notes for understanding:
# - Linear regression predicts a continuous number (assumes linearity, normal
#   residuals, homoscedasticity). Not suitable for binary adoption.
# - Logistic regression predicts a probability in [0,1] and classifies via a
#   threshold—perfect for binary "adopted vs not".
# - We one-hot-encode categoricals, standardize numerics, and validate on a
#   holdout test set (train_test_split).
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)

# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
SAMPLE_LIMIT = int(os.getenv("SAMPLE_LIMIT", "1000000"))   # rows per table
RANDOM_STATE = 1

TURNOVER_ORDER = ["< R5m", "R5m-R20m", "R20m-R100m", "R100m+"]
BEHAV_NUM_COLS = [
    "avg_balance", "inflow_txn_cnt", "outflow_txn_cnt",
    "inflow_amount", "outflow_amount",
    "digital_logins_cnt", "days_active", "avg_ticket_size",
    "email_open_rate", "email_ctr"
]

# ------------------ HELPERS ------------------

# Create a Supabase client
# Why: allows us to connect to our Supabase database and query tables
def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch a sample of rows from a table
# Why: we don’t want to load millions of rows for testing, sampling keeps it fast
def fetch_sample(sb: Client, table: str, limit: int = SAMPLE_LIMIT) -> pd.DataFrame:
    resp = sb.table(table).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

# Try fetching dim_client table (handles plural naming differences)
# Why: schema might use “dim_client” or “dim_clients”, this avoids errors
def try_fetch_dim_client(sb: Client) -> pd.DataFrame:
    for name in ("dim_client", "dim_clients"):
        try:
            df = fetch_sample(sb, name)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

# Convert surrogate key columns (*_sk) to numeric
# Why: ensures joins/group-bys work correctly, avoids string mismatches
def coerce_sk_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c.endswith("_sk"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Force selected columns to numeric
# Why: behaviour columns sometimes load as strings, need numeric for math/stats
def to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# Compute McFadden’s pseudo-R² for logistic regression
# Why: logistic doesn’t have standard R², pseudo-R² is a proxy for model fit
def mcfadden_pseudo_r2(y_true: np.ndarray, p_full: np.ndarray) -> float:
    eps = 1e-12
    p_full = np.clip(p_full, eps, 1 - eps)
    ll_full = np.sum(y_true * np.log(p_full) + (1 - y_true) * np.log(1 - p_full))
    base = np.clip(np.full_like(y_true, y_true.mean()), eps, 1 - eps)
    ll_null = np.sum(y_true * np.log(base) + (1 - y_true) * np.log(1 - base))
    return float(1 - (ll_full / ll_null)) if ll_null != 0 else np.nan

# ------------------ MAIN ------------------
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)

    # 1) LOAD DATA ------------------------------------------------------------
    # Why: load required tables (clients, industries, monthly behaviour, product adoption)
    sb = get_client()
    print("✅ Connected to Supabase")

    dim_client = try_fetch_dim_client(sb)
    dim_industry = fetch_sample(sb, "dim_industry")
    fcm = fetch_sample(sb, "fact_client_monthly")        # behaviour (monthly)
    fpa = fetch_sample(sb, "fact_product_adoption")      # adoption flags

    # Guard clause if required data is missing
    if dim_client.empty or fcm.empty or fpa.empty:
        raise SystemExit("Required tables returned no data (dim_client(s), fact_client_monthly, fact_product_adoption).")

    # Ensure surrogate keys are numeric for joins
    for df in (dim_client, dim_industry, fcm, fpa):
        coerce_sk_numeric(df)

    # Order turnover bands logically for plots/interpretation
    if "turnover_band" in dim_client.columns:
        dim_client["turnover_band"] = pd.Categorical(dim_client["turnover_band"],
                                                     categories=TURNOVER_ORDER, ordered=True)

    # 2) BUILD CLIENT-LEVEL DATASET ------------------------------------------
    # Target (y): client adopted any product? (1 if yes)
    target = (
        fpa.groupby("client_sk", as_index=False)["adopted_flag"]
           .max()
           .rename(columns={"adopted_flag": "adopted_any"})
    )

    # Aggregate monthly behaviour to client level (mean values per client)
    fcm = to_numeric(fcm, BEHAV_NUM_COLS)
    beh_agg = (
        fcm.groupby("client_sk", as_index=False)[BEHAV_NUM_COLS].mean()
    )

    # Select client profile features
    profile_cols = ["client_sk", "industry_sk", "segment", "turnover_band", "years_in_business", "employees_band"]
    profile = dim_client[[c for c in profile_cols if c in dim_client.columns]].copy()

    # Bring in industry_name for interpretability
    if not dim_industry.empty and {"industry_sk","industry_name"}.issubset(dim_industry.columns):
        profile = profile.merge(dim_industry[["industry_sk", "industry_name"]], on="industry_sk", how="left")

    # Merge target + profile + behaviour into one dataset
    data = (
        target.merge(profile, on="client_sk", how="inner")
              .merge(beh_agg, on="client_sk", how="inner")
    )
    
    print(data.shape)

    # Clean target dtype
    data["adopted_any"] = data["adopted_any"].fillna(False).astype(int)

    # 3) FEATURES & ENCODING --------------------------------------------------
    # Split features into numeric and categorical
    numeric_features = [c for c in BEHAV_NUM_COLS if c in data.columns] + ["years_in_business"]
    categorical_features = [c for c in ["industry_name", "segment", "turnover_band", "employees_band"] if c in data.columns]

    if len(numeric_features) == 0 and len(categorical_features) == 0:
        raise SystemExit("No features found to train on.")

    X = data[numeric_features + categorical_features].copy()
    y = data["adopted_any"].copy()

    # Preprocessing: scale numeric, one-hot encode categoricals
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop"
    )

    # Model: Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])

    # 4) VALIDATION: Train/Test Split -----------------------------------------
    # Why: validate model on unseen data to avoid overfitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipe.fit(X_train, y_train)

    # Predictions / Probabilities
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # Metrics (classification)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)
    pseudo_r2 = mcfadden_pseudo_r2(y_test.values, y_prob)

    print("\n=== Performance (Test Set) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"ROC AUC  : {roc:.3f}")
    print(f"McFadden pseudo-R²: {pseudo_r2:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion Matrix for errors vs correct predictions
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.show()

    # ROC Curve for trade-off visualization
    RocCurveDisplay.from_estimator(pipe, X_test, y_test)
    plt.title("ROC Curve (Test)")
    plt.show()

    # 5) DRIVER ANALYSIS: Coefficients ---------------------------------------
    # Extract feature names after preprocessing
    prep: ColumnTransformer = pipe.named_steps["prep"]
    clf: LogisticRegression = pipe.named_steps["clf"]

    num_names = numeric_features
    cat_names = []
    if len(categorical_features) > 0:
        ohe: OneHotEncoder = prep.named_transformers_["cat"]
        ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
        cat_names = ohe_names

    feature_names = num_names + cat_names
    coefs = clf.coef_.ravel()

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top_drivers = coef_df.sort_values("abs_coef", ascending=False).head(15)

    print("\n=== Top drivers (absolute coefficient, standardized) ===")
    print(top_drivers[["feature", "coef"]])

    # Visualize drivers for interpretability
    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top_drivers.sort_values("coef"),
        x="coef", y="feature", orient="h", palette="coolwarm"
    )
    plt.title("Top Logistic Regression Drivers (Std. Coefficients)")
    plt.xlabel("Coefficient (positive → higher adoption odds)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # 6) SCORING & SEGMENTATION ----------------------------------------------
    # Score ALL rows to generate adoption probabilities
    all_probs = pipe.predict_proba(X)[:, 1]
    data["adoption_prob"] = all_probs

    # Segment into High / Medium / Low
    q20, q80 = np.quantile(all_probs, [0.2, 0.8])
    def segment(p):
        if p >= q80: return "High"
        if p < q20:  return "Low"
        return "Medium"
    data["adoption_segment"] = data["adoption_prob"].apply(segment)

    print("\n=== Segments (counts) ===")
    print(data["adoption_segment"].value_counts())

    # Visualise probability distributions by segment
    plt.figure(figsize=(8,4))
    sns.kdeplot(data=data, x="adoption_prob", hue="adoption_segment", common_norm=False)
    plt.title("Predicted Adoption Probability by Segment")
    plt.xlabel("Predicted probability")
    plt.show()

    # Check adoption rate per segment (business-friendly summary)
    seg_rates = data.groupby("adoption_segment")["adopted_any"].mean().sort_index()
    print("\nObserved adoption rate by segment:")
    print((seg_rates * 100).round(1).astype(str) + "%")