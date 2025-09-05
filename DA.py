# descriptive_analysis.py
# ==========================================================
# Purpose:
#   1) Clean & validate data (missing, duplicates, numeric, outliers, categoricals)
#   2) Descriptive analysis on:
#        - dim_clients
#        - fact_client_monthly
#        - fact_transactions_daily
#   3) Save reporting outputs back to Supabase (public schema):
#        - dim_client_report
#        - fact_client_monthly_report
#        - fact_transactions_daily_report
# ==========================================================

import os
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from supabase import create_client

# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
LIMIT = int(os.getenv("SAMPLE_LIMIT", "1000000"))  # sample rows for quick analysis

# ------------------ BASIC HELPERS ------------------
def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_sample(supabase, table_name, limit=LIMIT):
    """Fetch sample rows from Supabase table"""
    resp = supabase.table(table_name).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

# ------------------ DATA PREP HELPERS ------------------
def check_missing(df, name):
    """Report columns with missing values (>0)"""
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"⚠️ {name}: Missing values\n{missing}")
    else:
        print(f"✅ {name}: No missing values")

def check_duplicates(df, name, subset=None):
    """Drop duplicates and report how many removed"""
    before = df.shape[0]
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    after = df.shape[0]
    dropped = before - after
    if dropped > 0:
        print(f"⚠️ {name}: {dropped} duplicate rows removed (subset={subset})")
    else:
        print(f"✅ {name}: No duplicates (subset={subset})")
    return df

def ensure_numeric(df, col, name):
    """Coerce a column to numeric, report NaNs produced"""
    if col in df.columns:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        delta = after - before
        if delta > 0:
            print(f"ℹ️ {name}.{col}: coerced to numeric; +{delta} NaNs from invalids")
        else:
            print(f"✅ {name}.{col}: numeric conversion OK")
    return df

def detect_outliers_iqr(df, col, name):
    """Flag outliers count via IQR rule for a numeric column; returns count"""
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        print(f"🔎 {name}.{col}: {n} outliers (IQR rule)")
        return n
    return 0

def validate_categorical(df, col, allowed, name):
    """Check that categorical values fall within allowed set"""
    if col in df.columns:
        invalid = set(df[col].dropna().unique()) - set(allowed)
        if invalid:
            print(f"⚠️ {name}.{col}: invalid categories {invalid}")
        else:
            print(f"✅ {name}.{col}: categories valid")
    return df

# ------------------ SAVE HELPERS (public schema) ------------------
def chunked_upsert_public(supabase, table: str, rows: list[dict], batch_size: int = 500):
    """Upsert to public.<table> in manageable batches."""
    if not rows:
        print(f"ℹ️ {table}: nothing to upsert.")
        return
    print(f"⬆️  Upserting {len(rows)} rows into {table} ...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        supabase.table(table).upsert(batch).execute()
    print(f"✅ Upsert complete for {table}")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sb = get_client()
    print("✅ Connected to Supabase")

    # For final cleaning summary
    summary = {
        "dim_client": {"rows": 0, "cols": 0, "dups_removed": 0, "missing_cells": 0, "outliers": {}},
        "fact_client_monthly": {"rows": 0, "cols": 0, "dups_removed": 0, "missing_cells": 0, "outliers": {}},
        "fact_transactions_daily": {"rows": 0, "cols": 0, "dups_removed": 0, "missing_cells": 0, "outliers": {}},
    }

    # ================== 1) dim_clients ==================
    dim_client = fetch_sample(sb, "dim_clients")
    print("\n=== DATA PREP: dim_client (sample) ===")
    check_missing(dim_client, "dim_client")
    before = dim_client.shape[0]
    dim_client = check_duplicates(dim_client, "dim_client", subset=["client_sk"])
    summary["dim_client"]["dups_removed"] = before - dim_client.shape[0]
    validate_categorical(dim_client, "segment", ["Micro","Small","Medium","Corporate"], "dim_client")
    validate_categorical(dim_client, "kyc_status", ["Verified","Pending","Review"], "dim_client")
    summary["dim_client"]["rows"], summary["dim_client"]["cols"] = dim_client.shape
    summary["dim_client"]["missing_cells"] = int(dim_client.isna().sum().sum())

    print("\n=== dim_client (sample) ===")
    print(dim_client.head())

    # Descriptive stats
    print("\nClient segments distribution:")
    print(dim_client["segment"].value_counts())
    print("\nKYC status distribution:")
    print(dim_client["kyc_status"].value_counts())

    # Plot segment distribution
    sns.countplot(data=dim_client, x="segment", order=dim_client["segment"].value_counts().index)
    plt.title("Client Segments Distribution")
    plt.show()

    # ================== 2) fact_client_monthly ==================
    fact_client_monthly = fetch_sample(sb, "fact_client_monthly")
    print("\n=== DATA PREP: fact_client_monthly (sample) ===")
    check_missing(fact_client_monthly, "fact_client_monthly")
    before = fact_client_monthly.shape[0]
    fact_client_monthly = check_duplicates(fact_client_monthly, "fact_client_monthly", subset=["client_sk","month_sk"])
    summary["fact_client_monthly"]["dups_removed"] = before - fact_client_monthly.shape[0]

    # Ensure numeric then detect outliers
    outlier_cols_monthly = ["avg_balance","inflow_txn_cnt","outflow_txn_cnt"]
    for col in outlier_cols_monthly:
        fact_client_monthly = ensure_numeric(fact_client_monthly, col, "fact_client_monthly")
        n_out = detect_outliers_iqr(fact_client_monthly, col, "fact_client_monthly")
        summary["fact_client_monthly"]["outliers"][col] = n_out

    summary["fact_client_monthly"]["rows"], summary["fact_client_monthly"]["cols"] = fact_client_monthly.shape
    summary["fact_client_monthly"]["missing_cells"] = int(fact_client_monthly.isna().sum().sum())

    print("\n=== fact_client_monthly (sample) ===")
    print(fact_client_monthly.head())

    # Basic stats on balances and transactions
    print("\nBalance statistics:")
    print(fact_client_monthly["avg_balance"].describe())
    print("\nInflow/Outflow Txn Count statistics:")
    print(fact_client_monthly[["inflow_txn_cnt","outflow_txn_cnt"]].describe())

    # Plot balances (x ticks every 10,000)
    sns.histplot(fact_client_monthly["avg_balance"], bins=30, kde=True)
    plt.title("Distribution of Average Balances")
    plt.xlabel("Average Balance")
    plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(10000))
    plt.show()

    # ================== 3) fact_transactions_daily ==================
    fact_txn_daily = fetch_sample(sb, "fact_transactions_daily")
    print("\n=== DATA PREP: fact_transactions_daily (sample) ===")
    check_missing(fact_txn_daily, "fact_transactions_daily")
    before = fact_txn_daily.shape[0]
    fact_txn_daily = check_duplicates(fact_txn_daily, "fact_transactions_daily", subset=["client_sk","date_sk"])
    summary["fact_transactions_daily"]["dups_removed"] = before - fact_txn_daily.shape[0]

    # Ensure numeric then detect outliers
    outlier_cols_daily = ["debit_amt","credit_amt"]
    for col in outlier_cols_daily:
        fact_txn_daily = ensure_numeric(fact_txn_daily, col, "fact_transactions_daily")
        n_out = detect_outliers_iqr(fact_txn_daily, col, "fact_transactions_daily")
        summary["fact_transactions_daily"]["outliers"][col] = n_out

    summary["fact_transactions_daily"]["rows"], summary["fact_transactions_daily"]["cols"] = fact_txn_daily.shape
    summary["fact_transactions_daily"]["missing_cells"] = int(fact_txn_daily.isna().sum().sum())

    print("\n=== fact_transactions_daily (sample) ===")
    print(fact_txn_daily.head())

    # Daily transaction volumes (sum per date)
    daily_vol = (
        fact_txn_daily.groupby("date_sk")[["debit_amt","credit_amt"]]
        .sum()
        .reset_index()
    )
    print("\nDaily transaction volume (first few days):")
    print(daily_vol.head())

    # Plot daily debit vs credit amounts
    plt.figure(figsize=(10,4))
    sns.lineplot(data=daily_vol, x="date_sk", y="debit_amt", label="Debit Amount")
    sns.lineplot(data=daily_vol, x="date_sk", y="credit_amt", label="Credit Amount")
    plt.title("Daily Debit vs Credit Amounts")
    plt.xlabel("Date (YYYYMMDD)")
    plt.ylabel("Amount")
    plt.legend()
    plt.show()

    # ------------------ FINAL CLEANING SUMMARY ------------------
    print("\n==================== CLEANING SUMMARY ====================")
    for tbl in ["dim_client", "fact_client_monthly", "fact_transactions_daily"]:
        info = summary[tbl]
        print(f"\n{tbl}:")
        print(f"  Rows x Cols       : {info['rows']} x {info['cols']}")
        print(f"  Duplicates dropped: {info['dups_removed']}")
        print(f"  Missing cells     : {info['missing_cells']}")
        if info["outliers"]:
            print("  Outliers (IQR):")
            for k, v in info["outliers"].items():
                print(f"    - {k}: {v}")
        else:
            print("  Outliers (IQR): n/a")
    print("==========================================================")

    # ================== BUILD REPORTING OUTPUTS ==================
    # 1) dim_client_report (segment & KYC breakdown)
    as_of_date = date.today()  # snapshot date
    total_clients = len(dim_client)
    if total_clients == 0:
        dim_client_report_rows = []
    else:
        grp = (
            dim_client.groupby(["segment", "kyc_status"])
            .size()
            .reset_index(name="client_count")
        )
        grp["pct_of_total"] = (grp["client_count"] / total_clients * 100).round(3)
        dim_client_report_rows = [
            {
                "as_of_date": str(as_of_date),
                "segment": str(row["segment"]) if pd.notna(row["segment"]) else "Unknown",
                "kyc_status": str(row["kyc_status"]) if pd.notna(row["kyc_status"]) else "Unknown",
                "client_count": int(row["client_count"]),
                "pct_of_total": float(row["pct_of_total"]),
            }
            for _, row in grp.iterrows()
        ]

    # 2) fact_client_monthly_report (descriptive stats for key metrics)
    as_of_month = date.today().replace(day=1)  # month snapshot
    metrics = ["avg_balance", "inflow_txn_cnt", "outflow_txn_cnt"]
    monthly_rows = []

    # Overall (segment=None) — if your table uses a generated segment_key('ALL'), that’s handled by DB
    for m in metrics:
        if m in fact_client_monthly.columns and pd.api.types.is_numeric_dtype(fact_client_monthly[m]):
            s = fact_client_monthly[m].dropna()
            if len(s) > 0:
                monthly_rows.append({
                    "as_of_month": str(as_of_month),
                    "metric": m,
                    "segment": None,  # NULL → DB may coalesce to 'ALL' via generated column if you defined it
                    "mean_value": float(s.mean()),
                    "median_value": float(s.median()),
                    "min_value": float(s.min()),
                    "max_value": float(s.max()),
                    "stddev_value": float(s.std()),
                })

    # Optional: per-segment stats — if segment not in monthly table, join from dim_client by client_sk
    if "segment" not in fact_client_monthly.columns:
        if {"client_sk", "segment"}.issubset(dim_client.columns) and "client_sk" in fact_client_monthly.columns:
            fact_client_monthly = fact_client_monthly.merge(
                dim_client[["client_sk", "segment"]],
                on="client_sk",
                how="left"
            )

    if "segment" in fact_client_monthly.columns:
        for seg in fact_client_monthly["segment"].dropna().unique().tolist():
            sub = fact_client_monthly[fact_client_monthly["segment"] == seg]
            for m in metrics:
                if m in sub.columns and pd.api.types.is_numeric_dtype(sub[m]):
                    s = sub[m].dropna()
                    if len(s) > 0:
                        monthly_rows.append({
                            "as_of_month": str(as_of_month),
                            "metric": m,
                            "segment": str(seg),
                            "mean_value": float(s.mean()),
                            "median_value": float(s.median()),
                            "min_value": float(s.min()),
                            "max_value": float(s.max()),
                            "stddev_value": float(s.std()),
                        })

    # 3) fact_transactions_daily_report (daily debit/credit + net)
    daily_vol = daily_vol.copy()
    daily_vol["txn_date"] = pd.to_datetime(
        daily_vol["date_sk"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    ).dt.date

    fact_txn_daily_rows = []
    for _, row in daily_vol.iterrows():
        if pd.isna(row["txn_date"]):
            continue
        debit = float(row.get("debit_amt", 0) or 0)
        credit = float(row.get("credit_amt", 0) or 0)
        fact_txn_daily_rows.append({
            "txn_date": str(row["txn_date"]),
            "total_debit_amt": debit,
            "total_credit_amt": credit,
            "net_flow": credit - debit,
        })

    # ================== UPSERT TO SUPABASE (PUBLIC) ==================
    try:
        chunked_upsert_public(sb, "dim_client_report", dim_client_report_rows)
        chunked_upsert_public(sb, "fact_client_monthly_report", monthly_rows)
        chunked_upsert_public(sb, "fact_transactions_daily_report", fact_txn_daily_rows)
        print("\n✅ Saved reporting outputs to Supabase (public).")
    except Exception as e:
        print(f"\n❌ Error while saving to Supabase: {e}")

    print("\n✅ Descriptive analysis complete (clean → analyze → save).")