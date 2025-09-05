/* =========================================================
   0) SCHEMA
   ========================================================= */
CREATE SCHEMA IF NOT EXISTS analytics;
SET search_path = analytics;

/* =========================================================
   1) DIMENSIONS (INDEPENDENT)
   ========================================================= */

/* 1.1 dim_time (day grain) */
CREATE TABLE IF NOT EXISTS dim_time (
  date_sk        INTEGER PRIMARY KEY,                -- yyyymmdd
  calendar_date  DATE NOT NULL UNIQUE,
  day_of_week    SMALLINT NOT NULL,                  -- 1=Mon .. 7=Sun
  week_of_year   SMALLINT NOT NULL,
  month_sk       INTEGER NOT NULL,                   -- yyyymm
  month_num      SMALLINT NOT NULL,                  -- 1..12
  month_name     VARCHAR(12) NOT NULL,
  quarter_num    SMALLINT NOT NULL,                  -- 1..4
  quarter_label  VARCHAR(6)  NOT NULL,               -- e.g., 'Q1'
  year_num       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dim_time_month_sk ON dim_time (month_sk);
CREATE INDEX IF NOT EXISTS idx_dim_time_year_qtr  ON dim_time (year_num, quarter_num);

/* 1.2 dim_month (month grain) */
CREATE TABLE IF NOT EXISTS dim_month (
  month_sk        INTEGER PRIMARY KEY,               -- yyyymm
  month_start     DATE NOT NULL,
  month_end       DATE NOT NULL,
  month_label     VARCHAR(7) NOT NULL,               -- 'YYYY-MM'
  quarter_num     SMALLINT NOT NULL,
  year_num        INTEGER NOT NULL
);

/* 1.3 dim_geography */
CREATE TABLE IF NOT EXISTS dim_geography (
  geography_sk    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  province        VARCHAR(100),
  region          VARCHAR(100)
);

/* 1.4 dim_industry */
CREATE TABLE IF NOT EXISTS dim_industry (
  industry_sk     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  industry_code   VARCHAR(32) UNIQUE,
  industry_name   VARCHAR(200) NOT NULL
);

/* 1.5 dim_product */
CREATE TABLE IF NOT EXISTS dim_product (
  product_sk      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  product_code    VARCHAR(64) UNIQUE NOT NULL,
  product_group   VARCHAR(50) NOT NULL,             -- e.g., 'merchant','forex','payroll','savings'
  is_credit_flag  BOOLEAN NOT NULL DEFAULT FALSE
);

/* 1.6 dim_channel */
CREATE TABLE IF NOT EXISTS dim_channel (
  channel_sk      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  channel_name    VARCHAR(64) UNIQUE NOT NULL       -- 'App','Internet','WhatsApp','Glia','Branch','Call-centre'
);

/* 1.7 dim_campaign */
CREATE TABLE IF NOT EXISTS dim_campaign (
  campaign_sk     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  campaign_code   VARCHAR(64) UNIQUE NOT NULL,
  campaign_name   VARCHAR(200) NOT NULL,
  start_date      DATE,
  end_date        DATE
);

/* 1.8 dim_client (SCD2-friendly, but independent) */
CREATE TABLE IF NOT EXISTS dim_client (
  client_sk            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  client_id            VARCHAR(64) NOT NULL,         -- natural/business key
  valid_from           DATE NOT NULL,
  valid_to             DATE NOT NULL,
  is_current           BOOLEAN NOT NULL DEFAULT TRUE,

  legal_name           VARCHAR(255),
  registration_no      VARCHAR(64),
  onboarding_date      DATE,
  years_in_business    INTEGER,
  employees_band       VARCHAR(20),
  turnover_band        VARCHAR(20),
  segment              VARCHAR(30),
  kyc_status           VARCHAR(20),

  relationship_manager_id VARCHAR(64),
  geography_sk         BIGINT,   -- no FK
  industry_sk          BIGINT    -- no FK
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_dim_client_bk_rng
  ON dim_client (client_id, valid_from, valid_to);

CREATE INDEX IF NOT EXISTS idx_dim_client_current
  ON dim_client (client_id) WHERE is_current = TRUE;

CREATE INDEX IF NOT EXISTS idx_dim_client_geo_ind
  ON dim_client (geography_sk, industry_sk);

/* =========================================================
   2) FACTS (INDEPENDENT)
   ========================================================= */

/* 2.1 fact_client_monthly (client × month) */
CREATE TABLE IF NOT EXISTS fact_client_monthly (
  client_sk                BIGINT NOT NULL,  -- no FK
  month_sk                 INTEGER NOT NULL, -- no FK

  avg_balance              NUMERIC(18,2),
  inflow_txn_cnt           INTEGER,
  outflow_txn_cnt          INTEGER,
  inflow_amount            NUMERIC(18,2),
  outflow_amount           NUMERIC(18,2),
  intl_txn_flag            BOOLEAN,

  digital_logins_cnt       INTEGER,
  self_service_usage_flag  BOOLEAN,
  avg_ticket_size          NUMERIC(18,2),
  days_active              SMALLINT,

  has_merchant             BOOLEAN,
  has_forex                BOOLEAN,
  has_payroll              BOOLEAN,
  has_savings              BOOLEAN,

  email_open_rate          NUMERIC(5,2),
  email_ctr                NUMERIC(5,2),
  last_campaign_sk         BIGINT,          -- no FK

  CONSTRAINT pk_fact_client_monthly PRIMARY KEY (client_sk, month_sk)
);

CREATE INDEX IF NOT EXISTS idx_fcm_flags
  ON fact_client_monthly (has_merchant, has_forex, has_payroll, has_savings);

CREATE INDEX IF NOT EXISTS idx_fcm_month
  ON fact_client_monthly (month_sk);

/* 2.2 fact_transactions_daily (client × day) */
CREATE TABLE IF NOT EXISTS fact_transactions_daily (
  client_sk         BIGINT NOT NULL,  -- no FK
  date_sk           INTEGER NOT NULL, -- no FK

  debit_cnt         INTEGER,
  debit_amt         NUMERIC(18,2),
  credit_cnt        INTEGER,
  credit_amt        NUMERIC(18,2),
  cross_border_cnt  INTEGER,
  cross_border_amt  NUMERIC(18,2),

  CONSTRAINT pk_fact_txn_daily PRIMARY KEY (client_sk, date_sk)
);

CREATE INDEX IF NOT EXISTS idx_ftd_date ON fact_transactions_daily (date_sk);

/* 2.3 fact_campaign_touch (client × campaign × date) */
CREATE TABLE IF NOT EXISTS fact_campaign_touch (
  client_sk        BIGINT NOT NULL,  -- no FK
  campaign_sk      BIGINT NOT NULL,  -- no FK
  date_sk          INTEGER NOT NULL, -- no FK

  delivered_flag   BOOLEAN,
  opened_flag      BOOLEAN,
  clicked_flag     BOOLEAN,
  responded_flag   BOOLEAN,

  CONSTRAINT pk_fact_campaign_touch PRIMARY KEY (client_sk, campaign_sk, date_sk)
);

CREATE INDEX IF NOT EXISTS idx_fct_flags ON fact_campaign_touch (opened_flag, clicked_flag, responded_flag);

/* 2.4 fact_digital_events (client × channel × date) */
CREATE TABLE IF NOT EXISTS fact_digital_events (
  client_sk         BIGINT NOT NULL,  -- no FK
  channel_sk        BIGINT NOT NULL,  -- no FK
  date_sk           INTEGER NOT NULL, -- no FK

  sessions          INTEGER,
  logins            INTEGER,
  features_used_cnt INTEGER,

  CONSTRAINT pk_fact_digital_events PRIMARY KEY (client_sk, channel_sk, date_sk)
);

CREATE INDEX IF NOT EXISTS idx_fde_date ON fact_digital_events (date_sk);

/* 2.5 fact_product_adoption (labels/outcomes) */
CREATE TABLE IF NOT EXISTS fact_product_adoption (
  client_sk              BIGINT NOT NULL,  -- no FK
  product_sk             BIGINT NOT NULL,  -- no FK
  adoption_window_start  DATE NOT NULL,
  adoption_window_end    DATE NOT NULL,

  adopted_flag           BOOLEAN NOT NULL,
  adoption_date          DATE,

  CONSTRAINT pk_fact_product_adoption
    PRIMARY KEY (client_sk, product_sk, adoption_window_start)
);

CREATE INDEX IF NOT EXISTS idx_fpa_adopted ON fact_product_adoption (adopted_flag);
CREATE INDEX IF NOT EXISTS idx_fpa_window  ON fact_product_adoption (adoption_window_start, adoption_window_end);

/* =========================================================
   3) FEATURE STORE (INDEPENDENT)
   ========================================================= */

CREATE TABLE IF NOT EXISTS feature_cross_sell_client_month (
  client_id                 VARCHAR(64) NOT NULL,
  as_of_month               INTEGER NOT NULL,              -- yyyymm

  /* Profile */
  years_in_business         INTEGER,
  employees_band            VARCHAR(20),
  turnover_band             VARCHAR(20),
  industry_code             VARCHAR(32),

  /* Behaviour (windowed) */
  avg_balance_3m            NUMERIC(18,2),
  avg_balance_6m            NUMERIC(18,2),
  inflow_txn_cnt_3m         INTEGER,
  outflow_txn_cnt_3m        INTEGER,
  intl_txn_flag_6m          BOOLEAN,
  digital_logins_3m         INTEGER,
  self_service_any_3m       BOOLEAN,

  /* Holdings (current) */
  has_merchant              BOOLEAN,
  has_forex                 BOOLEAN,
  has_payroll               BOOLEAN,
  has_savings               BOOLEAN,

  /* Engagement */
  email_open_rate_3m        NUMERIC(5,2),
  email_ctr_3m              NUMERIC(5,2),
  last_campaign_response_3m BOOLEAN,

  /* Target (training only; null in serving) */
  adopted_noncredit_12m     BOOLEAN,
  adoption_date             DATE,

  CONSTRAINT pk_feature_cs_cm PRIMARY KEY (client_id, as_of_month)
);

CREATE INDEX IF NOT EXISTS idx_feat_asof ON feature_cross_sell_client_month (as_of_month);

/* =========================================================
   4) OPTIONAL: TIME SEEDERS (STILL INDEPENDENT)
   ========================================================= */

/* Seed dim_month for 2018-01..2030-12 */
WITH months AS (
  SELECT
    to_char(d::date, 'YYYYMM')::int AS month_sk,
    date_trunc('month', d)::date     AS month_start,
    (date_trunc('month', d) + INTERVAL '1 month - 1 day')::date AS month_end,
    to_char(d, 'YYYY-MM')            AS month_label,
    EXTRACT(quarter FROM d)::int     AS quarter_num,
    EXTRACT(year FROM d)::int        AS year_num
  FROM generate_series('2018-01-01'::date, '2030-12-01'::date, interval '1 month') d
)
INSERT INTO dim_month (month_sk, month_start, month_end, month_label, quarter_num, year_num)
SELECT m.*
FROM months m
ON CONFLICT (month_sk) DO NOTHING;

/* Seed dim_time using the span covered by dim_month */
WITH days AS (
  SELECT generate_series(min(month_start), max(month_end), interval '1 day')::date AS d
  FROM dim_month
)
INSERT INTO dim_time (
  date_sk, calendar_date, day_of_week, week_of_year, month_sk,
  month_num, month_name, quarter_num, quarter_label, year_num
)
SELECT
  to_char(d, 'YYYYMMDD')::int            AS date_sk,
  d                                      AS calendar_date,
  EXTRACT(isodow FROM d)::int            AS day_of_week,
  EXTRACT(week   FROM d)::int            AS week_of_year,
  to_char(d, 'YYYYMM')::int              AS month_sk,
  EXTRACT(month  FROM d)::int            AS month_num,
  to_char(d, 'Mon')                      AS month_name,
  EXTRACT(quarter FROM d)::int           AS quarter_num,
  'Q' || EXTRACT(quarter FROM d)::int    AS quarter_label,
  EXTRACT(year   FROM d)::int            AS year_num
FROM days
ON CONFLICT (date_sk) DO NOTHING;