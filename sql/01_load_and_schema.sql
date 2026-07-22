-- 01_load_and_schema.sql
-- Run in BigQuery after uploading data/subscribers.csv and
-- data/viewing_activity.csv to a GCS bucket (gs://<bucket>/streamsight/raw/).
--
-- Usage (bq CLI):
--   bq mk --dataset streamsight_raw
--   bq load --autodetect --source_format=CSV \
--     streamsight_raw.subscribers gs://<bucket>/streamsight/raw/subscribers.csv
--   bq load --autodetect --source_format=CSV \
--     streamsight_raw.viewing_activity gs://<bucket>/streamsight/raw/viewing_activity.csv

CREATE SCHEMA IF NOT EXISTS `streamsight_raw`
  OPTIONS (location = 'europe-west2');   -- London region, matches Sky's UK data residency

CREATE SCHEMA IF NOT EXISTS `streamsight_features`
  OPTIONS (location = 'europe-west2');

CREATE TABLE IF NOT EXISTS `streamsight_raw.subscribers` (
  account_id                     INT64,
  tenure_months                  INT64,
  plan                           STRING,
  base_monthly_price_gbp         FLOAT64,
  current_monthly_price_gbp      FLOAT64,
  n_price_rises_since_joining    INT64,
  months_since_last_price_rise   INT64,
  region                         STRING,
  primary_device                 STRING,
  num_devices_registered         INT64,
  household_size_est             INT64,
  contract_type                  STRING,
  months_to_contract_end         INT64,
  support_contacts_last_90d      INT64,
  billing_dispute_last_90d       INT64,
  outage_reported_last_90d       INT64,
  nps_last_survey                INT64,
  competitor_ad_exposure_score   FLOAT64,
  churn_probability_true         FLOAT64,   -- ground truth (simulation only; excluded from training views)
  churned_next_60d               INT64,
  upsell_probability_true        FLOAT64,   -- ground truth (simulation only; excluded from training views)
  accepted_upsell_offer          INT64
)
PARTITION BY RANGE_BUCKET(account_id, GENERATE_ARRAY(0, 50000, 5000));

CREATE TABLE IF NOT EXISTS `streamsight_raw.viewing_activity` (
  account_id             INT64,
  months_ago             INT64,
  hrs_sport              FLOAT64,
  hrs_movies             FLOAT64,
  hrs_drama_entertainment FLOAT64,
  hrs_kids               FLOAT64,
  hrs_documentary        FLOAT64,
  hrs_news               FLOAT64,
  hrs_reality            FLOAT64,
  total_viewing_hours    FLOAT64
);
