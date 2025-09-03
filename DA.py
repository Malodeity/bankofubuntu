# descriptive_analysis.py
# ==========================================================
# Purpose:
#   Run descriptive analysis on a few key tables from Supabase
#   - dim_client
#   - fact_client_monthly
#   - fact_transactions_daily
#
# Focus:
#   - Summarize client demographics
#   - Summarize balances & transaction behavior
#   - Show daily transaction trends
# ==========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from supabase import create_client
import matplotlib.ticker as mtick


# ------------------ CONFIG ------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
LIMIT = 1000  # sample rows for quick analysis

# ------------------ HELPER ------------------
def get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_sample(supabase, table_name, limit=LIMIT):
    """Fetch sample rows from Supabase table"""
    resp = supabase.table(table_name).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

# ------------------ MAIN ------------------
if __name__ == "__main__":
    sb = get_client()
    print("✅ Connected to Supabase")

    # 1) dim_client --------------------------
    dim_client = fetch_sample(sb, "dim_clients")
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

    # 2) fact_client_monthly -----------------
    fact_client_monthly = fetch_sample(sb, "fact_client_monthly")
    print("\n=== fact_client_monthly (sample) ===")
    print(fact_client_monthly.head())

    # Basic stats on balances and transactions
    print("\nBalance statistics:")
    print(fact_client_monthly["avg_balance"].describe())

    print("\nInflow/Outflow Txn Count statistics:")
    print(fact_client_monthly[["inflow_txn_cnt","outflow_txn_cnt"]].describe())

    # Plot balances
    sns.histplot(fact_client_monthly["avg_balance"], bins=30, kde=True)
    plt.title("Distribution of Average Balances")
    plt.xlabel("Average Balance")
    plt.gca().xaxis.set_major_locator(mtick.MultipleLocator(10000))
    plt.show()

    # 3) fact_transactions_daily -------------
    fact_txn_daily = fetch_sample(sb, "fact_transactions_daily")
    print("\n=== fact_transactions_daily (sample) ===")
    print(fact_txn_daily.head())

    # Daily transaction volumes
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

    print("\n✅ Descriptive analysis complete (sample tables).")