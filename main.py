"""
Generate fake data for a South African bank star-schema-like dataset (independent tables).
Outputs CSV files for each table into an output directory.

Tables:
- dim_time
- dim_month
- dim_geography
- dim_industry
- dim_product
- dim_channel
- dim_campaign
- dim_client
- fact_client_monthly           (forced to 1,000,000 rows)
- fact_transactions_daily       (forced to 1,000,000 rows)
- fact_campaign_touch           (forced to 1,000,000 rows)
- fact_digital_events           (forced to 1,000,000 rows)
- fact_product_adoption         (forced to 1,000,000 rows)
- feature_cross_sell_client_month

Author: ChatGPT (Ubuntu Bank of South Africa demo data)
"""

import os
import zipfile
import math
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ------------------------------
# Parameters (tweak volumes here)
# ------------------------------
BANK_NAME = "Ubuntu Bank of South Africa"
OUT_DIR = "/Users/malo/Desktop/capitec/output"

random.seed(42)
np.random.seed(42)

N_CLIENTS = 600                 # number of clients
MONTHS_BACK = 18                # how many months of data ending this month
DAILY_ACTIVITY_RATE = 0.25      # fraction of days per client with any transactions/events
N_CAMPAIGNS = 8                 # number of campaigns
TXN_SCALE = 1.0                 # multiplier to scale transaction volumes
DIGI_SCALE = 1.0                # multiplier to scale digital events

# ------------------------------
# South Africa geography (province -> key cities)
# ------------------------------
SA_GEO = {
    "Gauteng": ["Johannesburg", "Pretoria", "Sandton", "Soweto"],
    "Western Cape": ["Cape Town", "Stellenbosch", "Paarl"],
    "KwaZulu-Natal": ["Durban", "Pietermaritzburg", "Umhlanga"],
    "Eastern Cape": ["Gqeberha", "East London"],
    "Free State": ["Bloemfontein", "Welkom"],
    "Limpopo": ["Polokwane", "Thohoyandou"],
    "Mpumalanga": ["Mbombela", "Emalahleni"],
    "North West": ["Rustenburg", "Mahikeng"],
    "Northern Cape": ["Kimberley", "Upington"]
}

INDUSTRIES = [
    ("A01", "Agriculture"),
    ("M01", "Manufacturing"),
    ("R01", "Retail & Wholesale"),
    ("F01", "Financial Services"),
    ("T01", "Transport & Logistics"),
    ("I01", "Information Technology"),
    ("C01", "Construction"),
    ("H01", "Healthcare"),
    ("E01", "Education"),
    ("HSP", "Hospitality & Tourism")
]

PRODUCTS = [
    ("MERCH", "merchant", False),
    ("FOREX", "forex", False),
    ("PAYRL", "payroll", False),
    ("SAVE",  "savings", False),
    ("OD",    "overdraft", True),
    ("TERM",  "term_loan", True)
]

CHANNELS = ["App", "Internet", "WhatsApp", "Glia", "Branch", "Call-centre"]

# ------------------------------------
# Helpers
# ------------------------------------
def month_yyyymm(dt):
    return int(dt.strftime("%Y%m"))

def date_yyyymmdd(dt):
    return int(dt.strftime("%Y%m%d"))

def month_range(end_dt, months_back):
    """Return list of (month_start_date) for each month from (end - months_back + 1) .. end"""
    months = []
    y = end_dt.year
    m = end_dt.month
    for i in range(months_back-1, -1, -1):
        year = y if m - i > 0 else y - 1 + ((m - i - 1) // 12)
        month = ((m - i - 1) % 12) + 1
        months.append(date(year, month, 1))
    return months

def days_in_month(dt):
    if dt.month == 12:
        next_month = date(dt.year + 1, 1, 1)
    else:
        next_month = date(dt.year, dt.month + 1, 1)
    delta = next_month - dt
    return [dt + timedelta(days=i) for i in range(delta.days)]

def sample_bool(p):
    return random.random() < p

def force_million_rows(df, target=1_000_000):
    """
    Ensure the dataframe has exactly 'target' rows by upsampling or downsampling.
    Keeps value distribution roughly intact via repetition + random sampling.
    """
    n = len(df)
    if n == 0:
        raise ValueError("Generated dataframe is empty; cannot force to 1,000,000 rows.")
    if n >= target:
        return df.sample(n=target, random_state=42).reset_index(drop=True)
    reps = math.ceil(target / n)
    df_big = pd.concat([df] * reps, ignore_index=True)
    return df_big.sample(n=target, random_state=42).reset_index(drop=True)

# ------------------------------------
# Build time dims (last MONTHS_BACK months up to current month)
# ------------------------------------
today = date.today().replace(day=1)  # current month start
months = month_range(today, MONTHS_BACK)  # list of month start dates

# dim_month
dim_month = []
for ms in months:
    if ms.month == 12:
        me = date(ms.year+1, 1, 1) - timedelta(days=1)
    else:
        me = date(ms.year, ms.month+1, 1) - timedelta(days=1)
    dim_month.append({
        "month_sk": month_yyyymm(ms),
        "month_start": ms.isoformat(),
        "month_end": me.isoformat(),
        "month_label": ms.strftime("%Y-%m"),
        "quarter_num": ((ms.month - 1)//3) + 1,
        "year_num": ms.year
    })
dim_month_df = pd.DataFrame(dim_month)

# dim_time (for the span of dim_month)
all_days = []
for ms in months:
    all_days.extend(days_in_month(ms))

dim_time = []
for d in all_days:
    dim_time.append({
        "date_sk": date_yyyymmdd(d),
        "calendar_date": d.isoformat(),
        "day_of_week": int(d.strftime("%u")),       # 1=Mon..7=Sun
        "week_of_year": int(d.strftime("%V")),
        "month_sk": int(d.strftime("%Y%m")),
        "month_num": d.month,
        "month_name": d.strftime("%b"),
        "quarter_num": ((d.month - 1)//3) + 1,
        "quarter_label": f"Q{((d.month - 1)//3) + 1}",
        "year_num": d.year
    })
dim_time_df = pd.DataFrame(dim_time)

# ------------------------------------
# Build simple dims (geography, industry, product, channel, campaign)
# ------------------------------------
# dim_geography: one row per (province, city)
geo_rows = []
for prov, cities in SA_GEO.items():
    for city in cities:
        geo_rows.append({"province": prov, "region": city})
dim_geography_df = pd.DataFrame(geo_rows).reset_index().rename(columns={"index": "geography_sk"})
dim_geography_df["geography_sk"] = dim_geography_df["geography_sk"] + 1  # start at 1

# dim_industry
dim_industry_df = pd.DataFrame([
    {"industry_code": code, "industry_name": name}
    for code, name in INDUSTRIES
]).reset_index().rename(columns={"index": "industry_sk"})
dim_industry_df["industry_sk"] = dim_industry_df["industry_sk"] + 1

# dim_product
dim_product_df = pd.DataFrame([
    {"product_code": code, "product_group": group, "is_credit_flag": credit}
    for code, group, credit in PRODUCTS
]).reset_index().rename(columns={"index": "product_sk"})
dim_product_df["product_sk"] = dim_product_df["product_sk"] + 1

# dim_channel
dim_channel_df = pd.DataFrame([{"channel_name": c} for c in CHANNELS]).reset_index().rename(columns={"index": "channel_sk"})
dim_channel_df["channel_sk"] = dim_channel_df["channel_sk"] + 1

# dim_campaign
campaigns = []
start_months = random.sample(months, k=min(len(months), N_CAMPAIGNS))
for i, sm in enumerate(start_months, start=1):
    em = sm + timedelta(days=28)
    campaigns.append({
        "campaign_sk": i,
        "campaign_code": f"CM{i:03d}",
        "campaign_name": f"Cross-sell Wave {i}",
        "start_date": sm.isoformat(),
        "end_date": (em if em.month == sm.month else date(sm.year, sm.month, 28)).isoformat()
    })
dim_campaign_df = pd.DataFrame(campaigns)

# ------------------------------------
# dim_client (SCD2-friendly but single current row per client)
# ------------------------------------
client_rows = []
for cid in range(1, N_CLIENTS + 1):
    # assign geography and industry
    g_row = dim_geography_df.sample(1).iloc[0]
    i_row = dim_industry_df.sample(1).iloc[0]

    # business stats
    weights = np.linspace(0.15, 0.85, 31)
    weights = weights / weights.sum()
    years_in_business = np.random.choice(range(0, 31), p=weights)
    employees = np.random.choice([1, 5, 10, 20, 50, 100, 250, 500], p=[0.05, 0.15, 0.20, 0.22, 0.18, 0.12, 0.06, 0.02])
    turnover = np.random.lognormal(mean=13, sigma=0.6)  # ~ Rands per year

    employees_band = (
        "1-5" if employees <= 5 else
        "6-20" if employees <= 20 else
        "21-50" if employees <= 50 else
        "51-100" if employees <= 100 else
        "101-250" if employees <= 250 else
        "250+"
    )
    turnover_band = (
        "< R5m" if turnover < 5e6 else
        "R5m-R20m" if turnover < 2e7 else
        "R20m-R100m" if turnover < 1e8 else
        "R100m+"
    )

    onboarding_date = months[0] - timedelta(days=random.randint(30, 5 * 365))

    client_rows.append({
        "client_sk": cid,
        "client_id": f"C{cid:06d}",
        "valid_from": (months[0]).isoformat(),
        "valid_to": (months[-1] + timedelta(days=1)).isoformat(),
        "is_current": True,
        "legal_name": f"Client {cid} (Pty) Ltd",
        "registration_no": f"REG{cid:07d}",
        "onboarding_date": onboarding_date.isoformat(),
        "years_in_business": years_in_business,
        "employees_band": employees_band,
        "turnover_band": turnover_band,
        "segment": np.random.choice(["Micro", "Small", "Medium", "Corporate"], p=[0.35, 0.35, 0.25, 0.05]),
        "kyc_status": np.random.choice(["Verified", "Pending", "Review"], p=[0.9, 0.07, 0.03]),
        "relationship_manager_id": f"RM{random.randint(1, 60):03d}",
        "geography_sk": int(g_row["geography_sk"]),
        "industry_sk": int(i_row["industry_sk"])
    })
dim_client_df = pd.DataFrame(client_rows)

# ------------------------------------
# Facts
# ------------------------------------
# 1) fact_client_monthly
fcm_rows = []
for _, c in dim_client_df.iterrows():
    # client product holdings baseline
    base_prob = {"merchant": 0.35, "forex": 0.20, "payroll": 0.30, "savings": 0.60}
    # tweak by industry (rough priors)
    ind = int(c["industry_sk"])
    if ind % 5 == 0:
        base_prob["forex"] += 0.1
    if ind % 3 == 0:
        base_prob["merchant"] += 0.1

    has_merch = sample_bool(base_prob["merchant"])
    has_forex = sample_bool(base_prob["forex"])
    has_payrl = sample_bool(base_prob["payroll"])
    has_save = sample_bool(base_prob["savings"])

    # monthly activity
    for ms in months:
        month_sk = int(ms.strftime("%Y%m"))
        active = sample_bool(0.8)  # chance the client is active this month
        avg_balance = max(0, np.random.normal(120000, 80000))
        inflow_cnt = int(np.random.poisson(30 if active else 5))
        outflow_cnt = int(np.random.poisson(28 if active else 4))
        inflow_amt = max(0, np.random.normal(350000, 120000) * (1.0 if active else 0.4))
        outflow_amt = max(0, np.random.normal(330000, 110000) * (1.0 if active else 0.4))
        intl_flag = sample_bool(0.15 if has_forex else 0.05)

        digi_logins = int(np.random.poisson(14 if active else 2) * DIGI_SCALE)
        self_service = digi_logins > 0 or sample_bool(0.05)

        email_open_rate = round(max(0, min(100, np.random.normal(25, 12))), 2)
        email_ctr = round(max(0, min(100, email_open_rate * np.random.uniform(0.1, 0.6))), 2)
        last_campaign_sk = np.random.choice(dim_campaign_df["campaign_sk"]) if sample_bool(0.3) else None

        fcm_rows.append({
            "client_sk": int(c["client_sk"]),
            "month_sk": month_sk,
            "avg_balance": round(avg_balance, 2),
            "inflow_txn_cnt": inflow_cnt,
            "outflow_txn_cnt": outflow_cnt,
            "inflow_amount": round(inflow_amt, 2),
            "outflow_amount": round(outflow_amt, 2),
            "intl_txn_flag": intl_flag,
            "digital_logins_cnt": digi_logins,
            "self_service_usage_flag": self_service,
            "avg_ticket_size": round((inflow_amt + outflow_amt) / max(1, inflow_cnt + outflow_cnt), 2),
            "days_active": int(np.random.poisson(14 if active else 3)),
            "has_merchant": has_merch,
            "has_forex": has_forex,
            "has_payroll": has_payrl,
            "has_savings": has_save,
            "email_open_rate": email_open_rate,
            "email_ctr": email_ctr,
            "last_campaign_sk": int(last_campaign_sk) if last_campaign_sk is not None else None
        })
fact_client_monthly_df = pd.DataFrame(fcm_rows)

# 2) fact_transactions_daily
ftd_rows = []
for _, c in dim_client_df.iterrows():
    active_days = [d for d in all_days if random.random() < DAILY_ACTIVITY_RATE]
    for d in active_days:
        debit_cnt = np.random.poisson(2 * TXN_SCALE)
        credit_cnt = np.random.poisson(2 * TXN_SCALE)
        ftd_rows.append({
            "client_sk": int(c["client_sk"]),
            "date_sk": int(d.strftime("%Y%m%d")),
            "debit_cnt": int(debit_cnt),
            "debit_amt": round(max(0, np.random.normal(8000, 6000) * (debit_cnt + 0.5)), 2),
            "credit_cnt": int(credit_cnt),
            "credit_amt": round(max(0, np.random.normal(7500, 5500) * (credit_cnt + 0.5)), 2),
            "cross_border_cnt": int(np.random.binomial(credit_cnt, 0.05)),
            "cross_border_amt": round(max(0, np.random.normal(12000, 9000) * (credit_cnt + 0.25)), 2)
        })
fact_transactions_daily_df = pd.DataFrame(ftd_rows)

# 3) fact_campaign_touch
fct_rows = []
for _, c in dim_client_df.iterrows():
    touched = random.sample(list(dim_campaign_df["campaign_sk"]), k=random.randint(0, len(dim_campaign_df) // 2))
    for camp in touched:
        ms = random.choice(months)
        date_sk = int(ms.strftime("%Y%m")) * 100 + random.randint(1, 28)
        opened = sample_bool(0.35)
        clicked = opened and sample_bool(0.4)
        responded = clicked and sample_bool(0.25)
        fct_rows.append({
            "client_sk": int(c["client_sk"]),
            "campaign_sk": int(camp),
            "date_sk": date_sk,
            "delivered_flag": True,
            "opened_flag": opened,
            "clicked_flag": clicked,
            "responded_flag": responded
        })
fact_campaign_touch_df = pd.DataFrame(fct_rows)

# 4) fact_digital_events
fde_rows = []
channel_ids = list(range(1, len(CHANNELS) + 1))
for _, c in dim_client_df.iterrows():
    digi_days = [d for d in all_days if random.random() < DAILY_ACTIVITY_RATE]
    for d in digi_days:
        for ch_id in random.sample(channel_ids, k=random.randint(1, min(3, len(channel_ids)))):  # up to 3 channels/day
            sessions = int(np.random.poisson(1.5 * DIGI_SCALE))
            logins = int(max(0, np.random.poisson(1.0 * DIGI_SCALE) - 0.2))
            features_used = int(np.random.poisson(2.0 * DIGI_SCALE))
            if sessions + logins + features_used == 0:
                continue
            fde_rows.append({
                "client_sk": int(c["client_sk"]),
                "channel_sk": int(ch_id),
                "date_sk": int(d.strftime("%Y%m%d")),
                "sessions": sessions,
                "logins": logins,
                "features_used_cnt": features_used
            })
fact_digital_events_df = pd.DataFrame(fde_rows)

# 5) fact_product_adoption
fpa_rows = []
product_ids = list(dim_product_df["product_sk"])
for _, c in dim_client_df.iterrows():
    non_credit = [pid for pid in product_ids if not dim_product_df.loc[dim_product_df["product_sk"] == pid, "is_credit_flag"].iloc[0]]
    adoptions = random.sample(non_credit, k=random.randint(0, 2))
    for pid in adoptions:
        win_start = random.choice(months)
        win_end = win_start + timedelta(days=180)
        adopted = sample_bool(0.6)
        adoption_date = None
        if adopted:
            d0 = win_start
            d1 = win_end
            delta_days = (d1 - d0).days
            adoption_date = d0 + timedelta(days=random.randint(0, max(1, delta_days)))
        fpa_rows.append({
            "client_sk": int(c["client_sk"]),
            "product_sk": int(pid),
            "adoption_window_start": win_start.isoformat(),
            "adoption_window_end": win_end.isoformat(),
            "adopted_flag": bool(adopted),
            "adoption_date": adoption_date.isoformat() if adoption_date else None
        })
fact_product_adoption_df = pd.DataFrame(fpa_rows)

# ------------------------------------
# 🔢 Force 1,000,000 rows on specified fact tables
# ------------------------------------
fact_client_monthly_df     = force_million_rows(fact_client_monthly_df)
fact_transactions_daily_df = force_million_rows(fact_transactions_daily_df)
fact_campaign_touch_df     = force_million_rows(fact_campaign_touch_df)
fact_digital_events_df     = force_million_rows(fact_digital_events_df)
fact_product_adoption_df   = force_million_rows(fact_product_adoption_df)

# ------------------------------------
# Feature table (derived, synthetic & independent)
# ------------------------------------
# We'll take last 6 months as_of_months and compute rough features using the monthly fact
as_of_months = sorted(dim_month_df["month_sk"].tolist())[-6:]

feat_rows = []
for cid, group in fact_client_monthly_df.groupby("client_sk"):
    client_id = f"C{int(cid):06d}"
    group_sorted = group.sort_values("month_sk")
    for m in as_of_months:
        hist3 = group_sorted[group_sorted["month_sk"] <= m].tail(3)
        hist6 = group_sorted[group_sorted["month_sk"] <= m].tail(6)

        avg_bal_3m = float(hist3["avg_balance"].mean()) if len(hist3) > 0 else None
        avg_bal_6m = float(hist6["avg_balance"].mean()) if len(hist6) > 0 else None
        inflow_3m = int(hist3["inflow_txn_cnt"].sum()) if len(hist3) > 0 else 0
        outflow_3m = int(hist3["outflow_txn_cnt"].sum()) if len(hist3) > 0 else 0
        intl6 = bool(hist6["intl_txn_flag"].any()) if len(hist6) > 0 else False
        digi3 = int(hist3["digital_logins_cnt"].sum()) if len(hist3) > 0 else 0
        self3 = bool(hist3["self_service_usage_flag"].any()) if len(hist3) > 0 else False

        row_at_m = group_sorted[group_sorted["month_sk"] == m]
        if row_at_m.empty:
            row_at_m = group_sorted[group_sorted["month_sk"] < m].tail(1)
        if row_at_m.empty:
            continue
        r = row_at_m.iloc[0]

        email_open_rate_3m = float(hist3["email_open_rate"].mean()) if len(hist3) > 0 else 0.0
        email_ctr_3m = float(hist3["email_ctr"].mean()) if len(hist3) > 0 else 0.0
        last_campaign_resp_3m = bool(random.random() < 0.2)

        feat_rows.append({
            "client_id": client_id,
            "as_of_month": int(m),
            "years_in_business": None,  # left None to keep independence
            "employees_band": None,
            "turnover_band": None,
            "industry_code": None,
            "avg_balance_3m": round(avg_bal_3m, 2) if avg_bal_3m is not None else None,
            "avg_balance_6m": round(avg_bal_6m, 2) if avg_bal_6m is not None else None,
            "inflow_txn_cnt_3m": inflow_3m,
            "outflow_txn_cnt_3m": outflow_3m,
            "intl_txn_flag_6m": intl6,
            "digital_logins_3m": digi3,
            "self_service_any_3m": self3,
            "has_merchant": bool(r["has_merchant"]),
            "has_forex": bool(r["has_forex"]),
            "has_payroll": bool(r["has_payroll"]),
            "has_savings": bool(r["has_savings"]),
            "email_open_rate_3m": round(email_open_rate_3m, 2),
            "email_ctr_3m": round(email_ctr_3m, 2),
            "last_campaign_response_3m": last_campaign_resp_3m,
            "adopted_noncredit_12m": None,
            "adoption_date": None
        })
feature_df = pd.DataFrame(feat_rows)

# ------------------------------------
# Write all CSVs
# ------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

def write_df(df, name):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    return path

paths = {}
paths["dim_time"] = write_df(dim_time_df, "dim_time")
paths["dim_month"] = write_df(dim_month_df, "dim_month")
paths["dim_geography"] = write_df(dim_geography_df, "dim_geography")
paths["dim_industry"] = write_df(dim_industry_df, "dim_industry")
paths["dim_product"] = write_df(dim_product_df, "dim_product")
paths["dim_channel"] = write_df(dim_channel_df, "dim_channel")
paths["dim_campaign"] = write_df(dim_campaign_df, "dim_campaign")
paths["dim_client"] = write_df(dim_client_df, "dim_client")
paths["fact_client_monthly"] = write_df(fact_client_monthly_df, "fact_client_monthly")
paths["fact_transactions_daily"] = write_df(fact_transactions_daily_df, "fact_transactions_daily")
paths["fact_campaign_touch"] = write_df(fact_campaign_touch_df, "fact_campaign_touch")
paths["fact_digital_events"] = write_df(fact_digital_events_df, "fact_digital_events")
paths["fact_product_adoption"] = write_df(fact_product_adoption_df, "fact_product_adoption")
paths["feature_cross_sell_client_month"] = write_df(feature_df, "feature_cross_sell_client_month")

# Bundle into a zip
zip_path = os.path.join(OUT_DIR, "sa_bank_fake_data_bundle.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for _, p in paths.items():
        z.write(p, arcname=os.path.basename(p))

# Summary print
print("=== Ubuntu Bank of South Africa — Synthetic Data ===")
print("Bank name:", BANK_NAME)
for k, v in paths.items():
    print(f"{k:35s} -> {os.path.basename(v)}")
print("ZIP bundle:", os.path.basename(zip_path))