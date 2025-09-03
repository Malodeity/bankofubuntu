"""
dbtest.py - Sample data preparation checks using Supabase.

Connect to Supabase, fetch sample rows from configured dimension and fact tables,
and perform data preparation checks:
1. Duplicate detection and removal
2. Missing value reporting
3. Numeric type conversion and NaN reporting
4. Categorical validation against allowed values
5. Outlier detection via IQR method

Prints a summary per table.
"""

import os, time, math, typing as t
import pandas as pd
import numpy as np
from supabase import create_client, Client

# ========== CONFIG ==========
# Load credentials from environment (or fallback to defaults)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bbkwerllrsqlezrzxqqf.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJia3dlcmxscnNxbGV6cnp4cXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTYyODM2NzEsImV4cCI6MjA3MTg1OTY3MX0.-s6W-R_fg0JnE_-CUqtA8i6SSjSIlFaqbVb3k6R85Kg")
BATCH_SIZE = 1000  # how many rows to fetch per request

# ========== TABLE CONFIGURATION ==========
# Tables to pull from Supabase with primary names and fallback aliases:
#   primary: main table name
#   aliases: alternate names to try if primary is missing
TABLES = [
    ("dim_client", ["dim_clients"]),
    ("dim_industry", ["dim_industries"]),
    ("dim_product", ["dim_products"]),
    ("dim_geography", ["dim_geographies"]),
    ("dim_channel", ["dim_channels"]),
    ("dim_campaign", ["dim_campaigns"]),
    ("dim_time", []),
    ("dim_month", []),
    ("fact_client_monthly", ["fact_clients_monthly"]),
    ("fact_transactions_daily", ["fact_transaction_daily", "transactions_daily"]),
    ("fact_campaign_touch", ["fact_campaign_touches"]),
    ("fact_digital_events", ["fact_digital_event", "digital_events"]),
    ("fact_product_adoption", ["fact_product_adoptions"]),
    ("feature_cross_sell_client_month", ["feature_cross_sell_client_months"]),
]

# ========== CATEGORICAL VALIDATION RULES ==========
# Defines allowed values for categorical columns per table:
#   key: table name
#   value: dict mapping column names to list of permitted categories
CAT_RULES = {
    "dim_client": {
        "segment": ["Micro", "Small", "Medium", "Corporate"],
        "kyc_status": ["Verified", "Pending", "Review"],
    },
    "dim_product": {
        "product_group": ["merchant", "forex", "payroll", "savings", "overdraft", "term_loan"],
    },
}

# ========== NUMERIC VALIDATION RULES ==========
# Specifies numeric columns to coerce and report NaNs per table:
#   key: table name
#   value: list of column names to convert to numeric
NUM_RULES = {
    "fact_client_monthly": ["avg_balance","inflow_amount","outflow_amount","days_active"],
    "fact_transactions_daily": ["debit_amt","credit_amt"],
    "fact_digital_events": ["sessions","logins","features_used_cnt"],
    "feature_cross_sell_client_month": ["avg_balance_3m","avg_balance_6m","digital_logins_3m"],
}

# ========== DUPLICATE CHECK RULES ==========
# Specifies primary key columns for duplicate detection and removal:
#   key: table name
#   value: list of columns used to identify duplicates
PK_RULES = {
    "dim_client": ["client_id"],
    "fact_client_monthly": ["client_sk","month_sk"],
    "fact_transactions_daily": ["client_sk","date_sk"],
    "fact_campaign_touch": ["client_sk","campaign_sk","date_sk"],
    "fact_digital_events": ["client_sk","channel_sk","date_sk"],
    "fact_product_adoption": ["client_sk","product_sk","adoption_window_start"],
    "feature_cross_sell_client_month": ["client_id","as_of_month"],
}

# ========== SUPABASE HELPER FUNCTIONS ==========
# Utility functions to establish connection and fetch sample rows.
def get_client() -> Client:
    """
    Establish a connection to Supabase.
    Returns:
        Client: Initialized Supabase client using URL and KEY.
    """
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_sample_rows(supabase: Client, table_name: str, limit: int = 1000) -> pd.DataFrame:
    """
    Fetch a limited set of rows from a Supabase table.
    Args:
        supabase (Client): Supabase client instance.
        table_name (str): Name of the table to query.
        limit (int): Maximum number of rows to retrieve.
    Returns:
        pd.DataFrame: DataFrame containing the fetched rows.
    """
    resp = supabase.table(table_name).select("*").limit(limit).execute()
    return pd.DataFrame(resp.data or [])

def fetch_with_aliases(supabase: Client, primary: str, aliases) -> tuple[str, pd.DataFrame]:
    """
    Attempt to fetch data from the primary table and any fallback aliases.
    Args:
        supabase (Client): Supabase client instance.
        primary (str): Primary table name to query.
        aliases (list): List of alternate table names to try.
    Returns:
        tuple[str, pd.DataFrame]: The table name that returned data and its DataFrame.
    """
    for name in (primary, *aliases):
        try:
            df = fetch_sample_rows(supabase, name)
            if not df.empty:
                return name, df
        except Exception:
            pass
    return primary, pd.DataFrame()

# ========== DATA PREP UTILITIES ==========
def report_missing(df: pd.DataFrame) -> pd.Series:
    """
    Report columns with missing values.
    Args:
        df (pd.DataFrame): DataFrame to analyze.
    Returns:
        pd.Series: Count of missing values per column where count > 0.
    """
    return df.isna().sum()[lambda x: x > 0]

def drop_dups(df: pd.DataFrame, keys: list) -> tuple[pd.DataFrame, int]:
    """
    Drop duplicate rows based on primary key columns.
    Args:
        df (pd.DataFrame): DataFrame to check.
        keys (list): List of column names to consider for duplication.
    Returns:
        tuple[pd.DataFrame, int]: Cleaned DataFrame and number of dropped duplicates.
    """
    if not keys:
        return df, 0
    dups = df.duplicated(subset=keys).sum()
    if dups:
        df = df.drop_duplicates(subset=keys).reset_index(drop=True)
    return df, int(dups)

def ensure_numeric(df: pd.DataFrame, cols: list) -> dict:
    """
    Coerce specified columns to numeric, reporting any conversion issues.
    Args:
        df (pd.DataFrame): DataFrame to process.
        cols (list): List of column names to convert.
    Returns:
        dict: Mapping of column names to count of NaNs introduced.
    """
    report = {}
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            report[c] = int(df[c].isna().sum())
    return report

def validate_cats(df: pd.DataFrame, rules: dict) -> dict:
    """
    Validate categorical columns against allowed values.
    Args:
        df (pd.DataFrame): DataFrame to validate.
        rules (dict): Mapping of column names to allowed value lists.
    Returns:
        dict: Mapping of columns to sets of invalid values.
    """
    bad = {}
    for col, allowed in rules.items():
        if col in df.columns:
            invalid = set(df[col].dropna().unique()) - set(allowed)
            if invalid:
                bad[col] = invalid
    return bad

def detect_outliers(df: pd.DataFrame, cols: list) -> dict:
    """
    Detect outliers in numeric columns using the IQR method.
    Args:
        df (pd.DataFrame): DataFrame to analyze.
        cols (list): List of numeric column names.
    Returns:
        dict: Mapping of columns to count of outlier values.
    """
    out = {}
    for c in cols:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
    return out

# ========== MAIN EXECUTION ==========
# Runs sample data prep checks for each configured table.
if __name__ == "__main__":
    sb = get_client()
    print("✅ Connected to Supabase")

    # Loop through each configured table and fetch sample data
    for primary, aliases in TABLES:
        table_name, df = fetch_with_aliases(sb, primary, aliases)
        if df.empty:
            print(f"⚠️ {primary}: no data found")
            continue

        print(f"\n=== {table_name} ({df.shape[0]} rows, {df.shape[1]} cols) ===")

        # -- Duplicate Detection and Removal
        df, dropped = drop_dups(df, PK_RULES.get(primary, []))
        if dropped: print(f"• dropped {dropped} duplicate rows")

        # -- Missing Value Reporting
        miss = report_missing(df)
        if not miss.empty: print("• missing values:\n", miss.to_string())

        # -- Numeric Conversion and Validation
        num_cols = NUM_RULES.get(primary, [])
        if num_cols:
            numeric_nulls = ensure_numeric(df, num_cols)
            if any(v > 0 for v in numeric_nulls.values()):
                print("• numeric conversion introduced NaNs:", numeric_nulls)

        # -- Categorical Value Validation
        cat_rules = CAT_RULES.get(primary, {})
        if cat_rules:
            invalid_cats = validate_cats(df, cat_rules)
            if invalid_cats: print("• invalid categories:", invalid_cats)

        # -- Outlier Detection
        if num_cols:
            outliers = detect_outliers(df, num_cols)
            flagged = {k:v for k,v in outliers.items() if v>0}
            if flagged: print("• outliers flagged:", flagged)

    print("\n✅ Data preparation (sample mode) complete.")