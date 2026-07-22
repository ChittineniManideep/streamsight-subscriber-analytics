-- 02_feature_engineering.sql
-- Builds the modelling feature view in BigQuery: genre-affinity ratios,
-- viewing trend (engagement momentum), and risk flags. This view is what
-- Vertex AI Pipelines reads from at training and batch-prediction time.

CREATE OR REPLACE VIEW `streamsight_features.viewing_features` AS
WITH pivoted AS (
  SELECT
    account_id,
    SUM(IF(months_ago = 0, total_viewing_hours, 0)) AS hrs_month_current,
    SUM(IF(months_ago = 1, total_viewing_hours, 0)) AS hrs_month_minus1,
    SUM(IF(months_ago = 2, total_viewing_hours, 0)) AS hrs_month_minus2,
    AVG(hrs_sport)               AS avg_hrs_sport,
    AVG(hrs_movies)              AS avg_hrs_movies,
    AVG(hrs_drama_entertainment) AS avg_hrs_drama_entertainment,
    AVG(hrs_kids)                AS avg_hrs_kids,
    AVG(hrs_documentary)         AS avg_hrs_documentary,
    AVG(hrs_news)                AS avg_hrs_news,
    AVG(hrs_reality)             AS avg_hrs_reality,
    AVG(total_viewing_hours)     AS avg_total_viewing_hours
  FROM `streamsight_raw.viewing_activity`
  GROUP BY account_id
)
SELECT
  account_id,
  avg_total_viewing_hours,
  -- Engagement momentum: negative = disengaging, a known early-warning churn signal
  SAFE_DIVIDE(hrs_month_current - hrs_month_minus2, NULLIF(hrs_month_minus2, 0)) AS viewing_trend_3mo,
  -- Genre affinity shares (content-affinity features, sum to ~1)
  SAFE_DIVIDE(avg_hrs_sport, NULLIF(avg_total_viewing_hours, 0))               AS affinity_sport,
  SAFE_DIVIDE(avg_hrs_movies, NULLIF(avg_total_viewing_hours, 0))              AS affinity_movies,
  SAFE_DIVIDE(avg_hrs_drama_entertainment, NULLIF(avg_total_viewing_hours, 0)) AS affinity_drama_entertainment,
  SAFE_DIVIDE(avg_hrs_kids, NULLIF(avg_total_viewing_hours, 0))                AS affinity_kids,
  SAFE_DIVIDE(avg_hrs_documentary, NULLIF(avg_total_viewing_hours, 0))         AS affinity_documentary,
  SAFE_DIVIDE(avg_hrs_news, NULLIF(avg_total_viewing_hours, 0))                AS affinity_news,
  SAFE_DIVIDE(avg_hrs_reality, NULLIF(avg_total_viewing_hours, 0))             AS affinity_reality
FROM pivoted;

CREATE OR REPLACE VIEW `streamsight_features.churn_training_view` AS
SELECT
  s.account_id,
  s.tenure_months,
  s.plan,
  s.current_monthly_price_gbp,
  s.n_price_rises_since_joining,
  s.months_since_last_price_rise,
  s.region,
  s.primary_device,
  s.num_devices_registered,
  s.household_size_est,
  s.contract_type,
  s.months_to_contract_end,
  s.support_contacts_last_90d,
  s.billing_dispute_last_90d,
  s.outage_reported_last_90d,
  s.nps_last_survey,
  s.competitor_ad_exposure_score,
  v.avg_total_viewing_hours,
  v.viewing_trend_3mo,
  v.affinity_sport,
  v.affinity_movies,
  v.affinity_drama_entertainment,
  v.affinity_kids,
  v.affinity_documentary,
  v.affinity_news,
  v.affinity_reality,
  s.churned_next_60d AS label
FROM `streamsight_raw.subscribers` s
JOIN `streamsight_features.viewing_features` v USING (account_id);

CREATE OR REPLACE VIEW `streamsight_features.upsell_training_view` AS
SELECT
  s.* EXCEPT (churned_next_60d, churn_probability_true, upsell_probability_true, accepted_upsell_offer),
  v.avg_total_viewing_hours,
  v.viewing_trend_3mo,
  v.affinity_sport,
  v.affinity_movies,
  v.affinity_drama_entertainment,
  v.affinity_kids,
  v.affinity_documentary,
  v.affinity_news,
  v.affinity_reality,
  s.accepted_upsell_offer AS label
FROM `streamsight_raw.subscribers` s
JOIN `streamsight_features.viewing_features` v USING (account_id);

-- Segment-level rollup used by the content-affinity clustering step and
-- by the Looker Studio dashboard (genre affinity by region/plan).
CREATE OR REPLACE VIEW `streamsight_features.segment_genre_affinity` AS
SELECT
  s.plan,
  s.region,
  ROUND(AVG(v.affinity_sport), 3)               AS avg_affinity_sport,
  ROUND(AVG(v.affinity_movies), 3)              AS avg_affinity_movies,
  ROUND(AVG(v.affinity_drama_entertainment), 3) AS avg_affinity_drama_entertainment,
  ROUND(AVG(v.affinity_kids), 3)                AS avg_affinity_kids,
  ROUND(AVG(v.affinity_documentary), 3)         AS avg_affinity_documentary,
  ROUND(AVG(v.affinity_news), 3)                AS avg_affinity_news,
  ROUND(AVG(v.affinity_reality), 3)             AS avg_affinity_reality,
  COUNT(*) AS n_accounts
FROM `streamsight_raw.subscribers` s
JOIN `streamsight_features.viewing_features` v USING (account_id)
GROUP BY plan, region;
