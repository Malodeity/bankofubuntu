# adoption_analysis.py
# ==========================================================
# Goal:
#   - Compare adopters vs non-adopters of products
#   - Explore correlations with client profile & behaviour
#   - Visualize adoption patterns by industry_name & turnover band
# ==========================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from supabase import create_client

# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
LIMIT = 2000  # sample size per table for quick analysis

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
    """Fetch up to `limit` rows from a Supabase table."""
    resp = sb.table(table).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

def try_fetch_dim_client(sb):
    """Try both 'dim_client' and 'dim_clients'."""
    for name in ("dim_client", "dim_clients"):
        try:
            df = fetch_sample(sb, name)
            if not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

def coerce_numeric(df, cols):
    """Coerce selected columns to numeric (ignore missing)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def coerce_sk_numeric(df):
    """Make *_sk columns numeric for reliable joins/groupbys."""
    for c in df.columns:
        if c.endswith("_sk"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sb = get_client()
    print("✅ Connected to Supabase")

    # Load data (samples)
    dim_client = try_fetch_dim_client(sb)
    fpa = fetch_sample(sb, "fact_product_adoption")
    fcm = fetch_sample(sb, "fact_client_monthly")
    dim_industry = fetch_sample(sb, "dim_industry")

    if dim_client.empty or fpa.empty or fcm.empty:
        raise SystemExit("One or more required tables returned no data: dim_client(s), fact_product_adoption, fact_client_monthly")

    # Ensure keys numeric
    for df in (dim_client, fpa, fcm, dim_industry):
        coerce_sk_numeric(df)

    # Normalize/clean fields
    if "turnover_band" in dim_client.columns:
        dim_client["turnover_band"] = pd.Categorical(dim_client["turnover_band"], categories=TURNOVER_ORDER, ordered=True)

    # Coerce behaviour numerics (avoid object dtype)
    fcm = coerce_numeric(fcm, BEHAVIOUR_NUM_COLS)

    # Merge adoption + client profile + behaviour + industry_name
    merged = (
        fpa.merge(dim_client, on="client_sk", how="left")
           .merge(fcm, on="client_sk", how="left", suffixes=("", "_monthly"))
    )
    if not dim_industry.empty and {"industry_sk","industry_name"}.issubset(dim_industry.columns):
        merged = merged.merge(dim_industry[["industry_sk","industry_name"]], on="industry_sk", how="left")

    # Ensure adopted_flag exists & boolean
    if "adopted_flag" not in merged.columns:
        merged["adopted_flag"] = False
    merged["adopted_flag"] = merged["adopted_flag"].fillna(False).astype(bool)

    # ---------------------------
    # 1) Overall adoption rate
    # ---------------------------
    adoption_rate = float(merged["adopted_flag"].mean())
    print(f"\nOverall adoption rate: {adoption_rate:.2%}")

    # ---------------------------
    # 2) Adoption by industry_name (not industry_sk)
    # ---------------------------
    if "industry_name" in merged.columns:
        adoption_by_industry = (
            merged.groupby("industry_name", dropna=False)["adopted_flag"]
                  .mean().reset_index()
                  .sort_values("adopted_flag", ascending=False)
        )
        print("\nAdoption rate by industry_name (top 10):")
        print(adoption_by_industry.head(10))

        plt.figure(figsize=(10,5))
        sns.barplot(
            data=adoption_by_industry,
            x="industry_name", y="adopted_flag",
            order=adoption_by_industry["industry_name"].tolist()
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Adoption rate")
        plt.xlabel("Industry")
        plt.title("Adoption Rate by Industry")
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ industry_name column missing—check dim_industry join.")

    # ---------------------------
    # 3) Adoption by turnover band (ordered)
    # ---------------------------
    if "turnover_band" in merged.columns:
        adoption_by_turnover = (
            merged.groupby("turnover_band", dropna=False)["adopted_flag"]
                  .mean().reset_index()
                  .sort_values("turnover_band")
        )
        print("\nAdoption rate by turnover_band:")
        print(adoption_by_turnover)

        plt.figure(figsize=(8,4))
        sns.barplot(
            data=adoption_by_turnover,
            x="turnover_band", y="adopted_flag",
            order=TURNOVER_ORDER if set(TURNOVER_ORDER) & set(adoption_by_turnover["turnover_band"]) else None
        )
        plt.ylabel("Adoption rate")
        plt.xlabel("Turnover band")
        plt.title("Adoption Rate by Turnover Band")
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ turnover_band missing from merged data.")

    # ---------------------------
    # 4) Behavioural comparison: adopters vs non-adopters
    # ---------------------------
    if "avg_balance" in merged.columns:
        adopters = merged.loc[merged["adopted_flag"] == True]
        non_adopters = merged.loc[merged["adopted_flag"] == False]

        print("\nAverage balance (mean):")
        print("  Adopters     :", f"{adopters['avg_balance'].mean(skipna=True):,.2f}")
        print("  Non-adopters :", f"{non_adopters['avg_balance'].mean(skipna=True):,.2f}")

        plt.figure(figsize=(6,4))
        sns.boxplot(data=merged, x="adopted_flag", y="avg_balance")
        plt.xlabel("Adopted new product")
        plt.ylabel("Average balance")
        plt.title("Average Balance: Adopters vs Non-Adopters")
        plt.tight_layout()
        plt.show()
    else:
        print("\n⚠️ avg_balance not available in merged data.")

    # ---------------------------
    # 5) Correlation heatmap: behaviour vs adoption
    # ---------------------------
    merged["adopted_flag_num"] = merged["adopted_flag"].astype(int)
    corr_cols = [c for c in BEHAVIOUR_NUM_COLS if c in merged.columns] + ["adopted_flag_num"]

    if len(corr_cols) >= 2:
        corr_df = merged[corr_cols].copy()
        corr_df = corr_df.dropna(axis=1, how="all")  # drop all-NaN columns
        if corr_df.shape[1] >= 2:
            corr = corr_df.corr(numeric_only=True)

            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            plt.title("Correlation: Behaviour vs Adoption")
            plt.tight_layout()
            plt.show()

            print("\nTop correlations with adoption (absolute):")
            if "adopted_flag_num" in corr.columns:
                corr_target = (corr["adopted_flag_num"]
                               .drop(labels=["adopted_flag_num"])
                               .abs().sort_values(ascending=False))
                print(corr_target.head(10))
            else:
                print("  (adopted_flag_num not in correlation matrix?)")
        else:
            print("\n⚠️ Not enough numeric columns for correlation.")
    else:
        print("\n⚠️ No behavioural numeric columns found for correlation.")

    print("\n✅ Adoption analysis complete.")