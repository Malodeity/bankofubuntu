# adoption_analysis.py (fixed: JSON-safe upserts)
# ==========================================================
# Clean -> Analyze -> Save (public tables)
# - Sanitizes all UPSERT rows: NaN/Inf -> None; NumPy -> Python types
# - Guards std/means to avoid NaN outputs
# ==========================================================

import os, math
from datetime import date
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from supabase import create_client

# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
LIMIT = 1000000

TURNOVER_ORDER = ["< R5m", "R5m-R20m", "R20m-R100m", "R100m+"]

BEHAVIOUR_NUM_COLS = [
    "avg_balance","inflow_txn_cnt","outflow_txn_cnt",
    "inflow_amount","outflow_amount",
    "digital_logins_cnt","days_active","avg_ticket_size",
    "email_open_rate","email_ctr"
]

# ------------------ HELPERS ------------------
def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_sample(sb, table, limit=LIMIT):
    resp = sb.table(table).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

def try_fetch_dim_client(sb):
    for name in ("dim_client", "dim_clients"):
        try:
            df = fetch_sample(sb, name)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def coerce_sk_numeric(df):
    for c in df.columns:
        if c.endswith("_sk"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---------- Data Prep ----------
def check_missing(df, name):
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(f"⚠️ {name}: Missing values\n{miss}") if not miss.empty else print(f"✅ {name}: No missing values")

def drop_duplicates_report(df, name, subset=None):
    before = len(df); df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    dropped = before - len(df)
    print(f"⚠️ {name}: {dropped} duplicate rows removed (subset={subset})") if dropped else print(f"✅ {name}: No duplicates (subset={subset})")
    return df

def detect_outliers_iqr(df, col, name):
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        print(f"🔎 {name}.{col}: {n} outliers (IQR rule)")
        return n
    return 0

def validate_categorical(df, col, allowed, name):
    if col not in df.columns: return
    invalid = set(df[col].dropna().unique()) - set(allowed)
    print(f"⚠️ {name}.{col}: invalid categories {invalid}") if invalid else print(f"✅ {name}.{col}: categories valid")

# ---------- JSON sanitization (fixes your error) ----------
def _to_jsonable(v):
    # None stays None
    if v is None: return None
    # Handle numpy types
    if isinstance(v, (np.floating,)):
        fv = float(v)
        return fv if math.isfinite(fv) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    # dates/strings/ints/bools are fine
    return v

def sanitize_rows(rows: list[dict]) -> list[dict]:
    clean = []
    for r in rows:
        clean.append({k: _to_jsonable(v) for k, v in r.items()})
    return clean

# ---------- Save helpers ----------
def upsert_public(sb, table: str, rows: list[dict], batch_size: int = 500):
    if not rows:
        print(f"ℹ️ {table}: nothing to upsert."); return
    rows = sanitize_rows(rows)  # <-- crucial: remove NaN/Inf, cast numpy
    print(f"⬆️  Upserting {len(rows)} rows into {table} ...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        sb.table(table).upsert(batch).execute()
    print(f"✅ Upsert complete for {table}")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sb = get_client()
    print("✅ Connected to Supabase")
    as_of_date = date.today()

    # LOAD
    dim_client   = try_fetch_dim_client(sb)
    fpa          = fetch_sample(sb, "fact_product_adoption")
    fcm          = fetch_sample(sb, "fact_client_monthly")
    dim_industry = fetch_sample(sb, "dim_industry")

    if dim_client.empty or fpa.empty or fcm.empty:
        raise SystemExit("One or more required tables returned no data: dim_client(s), fact_product_adoption, fact_client_monthly")

    # PREP
    for df_name, df in [("dim_client", dim_client), ("fact_product_adoption", fpa), ("fact_client_monthly", fcm), ("dim_industry", dim_industry)]:
        coerce_sk_numeric(df)
        check_missing(df, df_name)
        if df_name == "dim_client":
            dim_client = drop_duplicates_report(df, df_name, subset=["client_sk"])
        elif df_name == "fact_product_adoption":
            fpa = drop_duplicates_report(df, df_name, subset=None)
        elif df_name == "fact_client_monthly":
            fcm = drop_duplicates_report(df, df_name, subset=["client_sk","month_sk"]) if {"client_sk","month_sk"}.issubset(df.columns) else drop_duplicates_report(df, df_name, subset=None)
        else:
            dim_industry = drop_duplicates_report(df, df_name, subset=["industry_sk"]) if "industry_sk" in df.columns else drop_duplicates_report(df, df_name, subset=None)

    if "turnover_band" in dim_client.columns:
        dim_client["turnover_band"] = pd.Categorical(dim_client["turnover_band"], categories=TURNOVER_ORDER, ordered=True)
        validate_categorical(dim_client, "turnover_band", TURNOVER_ORDER, "dim_client")

    fcm = coerce_numeric(fcm, BEHAVIOUR_NUM_COLS)
    for col in ["avg_balance","inflow_txn_cnt","outflow_txn_cnt","digital_logins_cnt"]:
        _ = detect_outliers_iqr(fcm, col, "fact_client_monthly")

    # MERGE
    merged = (
        fpa.merge(dim_client, on="client_sk", how="left")
           .merge(fcm, on="client_sk", how="left", suffixes=("", "_monthly"))
    )
    if not dim_industry.empty and {"industry_sk","industry_name"}.issubset(dim_industry.columns):
        merged = merged.merge(dim_industry[["industry_sk","industry_name"]], on="industry_sk", how="left")

    if "adopted_flag" not in merged.columns:
        merged["adopted_flag"] = False
    merged["adopted_flag"] = merged["adopted_flag"].fillna(False).astype(bool)

    # 1) Overall
    adoption_rate = float(merged["adopted_flag"].mean())
    if not math.isfinite(adoption_rate): adoption_rate = 0.0
    print(f"\nOverall adoption rate: {adoption_rate:.2%}")

    # 2) By industry
    adoption_by_industry_rows = []
    if "industry_name" in merged.columns:
        adoption_by_industry = (
            merged.groupby("industry_name", dropna=False)["adopted_flag"]
                  .agg(["mean","count"])
                  .reset_index()
                  .rename(columns={"mean":"adoption_rate","count":"sample_size"})
                  .sort_values("adoption_rate", ascending=False)
        )
        for _, r in adoption_by_industry.iterrows():
            industry_val = "Unknown" if pd.isna(r["industry_name"]) else str(r["industry_name"])
            rate = float(r["adoption_rate"]);  rate = rate if math.isfinite(rate) else 0.0
            adoption_by_industry_rows.append({
                "as_of_date": str(as_of_date),
                "industry_name": industry_val,
                "adoption_rate": rate,
                "sample_size": int(r["sample_size"]),
            })

        # (plots omitted in headless runs)
        # sns.barplot(...)

    # 3) By turnover band
    adoption_by_turnover_rows = []
    if "turnover_band" in merged.columns:
        adoption_by_turnover = (
            merged.groupby("turnover_band", dropna=False)["adopted_flag"]
                  .agg(["mean","count"]).reset_index()
                  .rename(columns={"mean":"adoption_rate","count":"sample_size"})
        )
        for _, r in adoption_by_turnover.iterrows():
            band_val = "UNKNOWN" if pd.isna(r["turnover_band"]) else str(r["turnover_band"])
            rate = float(r["adoption_rate"]);  rate = rate if math.isfinite(rate) else 0.0
            adoption_by_turnover_rows.append({
                "as_of_date": str(as_of_date),
                "turnover_band": band_val,
                "adoption_rate": rate,
                "sample_size": int(r["sample_size"]),
            })

    # 4) Behavioural comparison
    balance_compare_rows = []
    if "avg_balance" in merged.columns:
        adopters = merged.loc[merged["adopted_flag"] == True, "avg_balance"].dropna()
        non_adopters = merged.loc[merged["adopted_flag"] == False, "avg_balance"].dropna()
        adop_mean = float(adopters.mean()) if len(adopters) else 0.0
        non_mean  = float(non_adopters.mean()) if len(non_adopters) else 0.0
        balance_compare_rows.append({
            "as_of_date": str(as_of_date),
            "metric": "avg_balance",
            "mean_adopters": adop_mean,
            "mean_non_adopters": non_mean
        })

    # 5) Correlations
    corr_rows = []
    merged["adopted_flag_num"] = merged["adopted_flag"].astype(int)
    corr_cols = [c for c in BEHAVIOUR_NUM_COLS if c in merged.columns] + ["adopted_flag_num"]
    if len(corr_cols) >= 2:
        corr_df = merged[corr_cols].dropna(axis=1, how="all")
        if corr_df.shape[1] >= 2:
            corr = corr_df.corr(numeric_only=True)
            if "adopted_flag_num" in corr.columns:
                corr_target = (corr["adopted_flag_num"]
                               .drop(labels=["adopted_flag_num"])
                               .sort_values(key=np.abs, ascending=False))
                for feat, val in corr_target.head(20).items():
                    if val is not None and math.isfinite(float(val)):
                        corr_rows.append({
                            "as_of_date": str(as_of_date),
                            "feature": str(feat),
                            "corr_with_adoption": float(val)
                        })

    # Build overall row
    overall_rows = [{"as_of_date": str(as_of_date), "adoption_rate": adoption_rate}]

    # SAVE (JSON-safe)
    try:
        upsert_public(sb, "adoption_overall_report", overall_rows)
        upsert_public(sb, "adoption_by_industry_report", adoption_by_industry_rows)
        upsert_public(sb, "adoption_by_turnover_report", adoption_by_turnover_rows)
        upsert_public(sb, "adoption_balance_comparison_report", balance_compare_rows)
        upsert_public(sb, "adoption_correlations_report", corr_rows)
        print("\n✅ Saved adoption analysis outputs to Supabase (public).")
    except Exception as e:
        print(f"\n❌ Error while saving to Supabase: {e}")

    print("\n✅ Adoption analysis complete.")