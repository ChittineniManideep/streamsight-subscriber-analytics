"""
features.py
-----------
Local (pandas) implementation of the same feature engineering logic defined
in sql/02_feature_engineering.sql. Keeping the two in lockstep means the
notebooks here can be validated locally, then the identical logic runs at
scale in BigQuery when this pipeline is deployed on GCP.
"""

import numpy as np
import pandas as pd

GENRES = ["sport", "movies", "drama_entertainment", "kids", "documentary", "news", "reality"]


def build_viewing_features(activity: pd.DataFrame) -> pd.DataFrame:
    pivot_hours = activity.pivot(index="account_id", columns="months_ago", values="total_viewing_hours")
    pivot_hours.columns = [f"hrs_month_minus{c}" for c in pivot_hours.columns]

    avg_by_genre = activity.groupby("account_id")[[f"hrs_{g}" for g in GENRES] + ["total_viewing_hours"]].mean()
    avg_by_genre = avg_by_genre.rename(columns={"total_viewing_hours": "avg_total_viewing_hours"})

    feats = pivot_hours.join(avg_by_genre)
    feats["viewing_trend_3mo"] = (
        (feats["hrs_month_minus0"] - feats["hrs_month_minus2"])
        / feats["hrs_month_minus2"].replace(0, np.nan)
    )
    for g in GENRES:
        feats[f"affinity_{g}"] = feats[f"hrs_{g}"] / feats["avg_total_viewing_hours"].replace(0, np.nan)

    keep = ["avg_total_viewing_hours", "viewing_trend_3mo"] + [f"affinity_{g}" for g in GENRES]
    return feats[keep].reset_index()


def build_churn_training_frame(subscribers: pd.DataFrame, viewing_features: pd.DataFrame) -> pd.DataFrame:
    df = subscribers.merge(viewing_features, on="account_id", how="inner")
    df["label"] = df["churned_next_60d"]
    drop_cols = ["churn_probability_true", "upsell_probability_true",
                 "accepted_upsell_offer", "churned_next_60d"]
    return df.drop(columns=drop_cols)


def build_upsell_training_frame(subscribers: pd.DataFrame, viewing_features: pd.DataFrame) -> pd.DataFrame:
    df = subscribers.merge(viewing_features, on="account_id", how="inner")
    df["label"] = df["accepted_upsell_offer"]
    drop_cols = ["churn_probability_true", "upsell_probability_true",
                 "accepted_upsell_offer", "churned_next_60d"]
    return df.drop(columns=drop_cols)


CATEGORICAL_COLS = ["plan", "region", "primary_device", "contract_type"]
NUMERIC_COLS = [
    "tenure_months", "current_monthly_price_gbp", "n_price_rises_since_joining",
    "months_since_last_price_rise", "num_devices_registered", "household_size_est",
    "months_to_contract_end", "support_contacts_last_90d", "billing_dispute_last_90d",
    "outage_reported_last_90d", "nps_last_survey", "competitor_ad_exposure_score",
    "avg_total_viewing_hours", "viewing_trend_3mo",
] + [f"affinity_{g}" for g in GENRES]
