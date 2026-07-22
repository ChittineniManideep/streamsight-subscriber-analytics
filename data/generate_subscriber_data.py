"""
generate_subscriber_data.py
----------------------------
Generates a synthetic subscription-media dataset that mimics the shape of
data a subscriber analytics team (e.g. Sky, Netflix, NOW) would work with:
account-level attributes, monthly viewing behaviour by genre, billing
events, support contacts, and churn/upgrade outcomes.

The dataset is entirely synthetic (no real customer data), but the feature
distributions and correlations are deliberately designed to reflect known
patterns in subscription-media churn literature (price sensitivity, low
engagement, recent price rises, and support friction all raise churn risk).

Output: data/subscribers.csv, data/viewing_activity.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SUBSCRIBERS = 50_000

GENRES = [
    "sport", "movies", "drama_entertainment", "kids",
    "documentary", "news", "reality"
]

PLANS = ["Basic TV", "Sky Signature", "Sky Sports", "Sky Cinema", "Ultimate TV"]
PLAN_BASE_PRICE = {
    "Basic TV": 15.0, "Sky Signature": 26.0, "Sky Sports": 46.0,
    "Sky Cinema": 36.0, "Ultimate TV": 65.0,
}
DEVICES = ["set_top_box", "sky_glass", "sky_stream", "mobile_app", "smart_tv_app"]
REGIONS = ["London", "South East", "North West", "Scotland", "Midlands", "South West", "Wales", "North East"]


def generate_subscribers(n=N_SUBSCRIBERS):
    account_id = np.arange(1, n + 1)
    tenure_months = RNG.gamma(shape=2.2, scale=14, size=n).clip(1, 180).astype(int)
    plan = RNG.choice(PLANS, size=n, p=[0.30, 0.28, 0.16, 0.14, 0.12])
    base_price = np.array([PLAN_BASE_PRICE[p] for p in plan])

    # Price rises: Sky (like most UK telco/media) runs annual price reviews.
    # Longer-tenure customers have absorbed more cumulative rises.
    n_price_rises = RNG.poisson(lam=tenure_months / 14).clip(0, 6)
    cumulative_price_rise_pct = n_price_rises * RNG.uniform(0.03, 0.09, size=n)
    current_price = base_price * (1 + cumulative_price_rise_pct)
    months_since_last_price_rise = RNG.integers(0, 15, size=n)

    region = RNG.choice(REGIONS, size=n)
    primary_device = RNG.choice(DEVICES, size=n, p=[0.22, 0.20, 0.18, 0.22, 0.18])
    num_devices_registered = RNG.integers(1, 6, size=n)
    household_size_est = RNG.integers(1, 6, size=n)

    contract_type = RNG.choice(["18mo_contract", "rolling_monthly"], size=n, p=[0.55, 0.45])
    # rolling monthly customers churn more easily and are past minimum term
    months_to_contract_end = np.where(
        contract_type == "18mo_contract",
        RNG.integers(-3, 19, size=n),  # can be negative = already out of contract
        0,
    )

    support_contacts_last_90d = RNG.poisson(lam=0.35, size=n)
    billing_dispute_last_90d = RNG.binomial(1, 0.06, size=n)
    outage_reported_last_90d = RNG.binomial(1, 0.09, size=n)
    nps_last_survey = RNG.integers(-2, 3, size=n) * 0 + RNG.integers(0, 11, size=n)
    nps_last_survey = np.clip(nps_last_survey - (support_contacts_last_90d * 1.2).astype(int), 0, 10)

    competitor_ad_exposure_score = RNG.uniform(0, 1, size=n)  # proxy: streaming/telco competitor ad exposure

    df = pd.DataFrame({
        "account_id": account_id,
        "tenure_months": tenure_months,
        "plan": plan,
        "base_monthly_price_gbp": base_price.round(2),
        "current_monthly_price_gbp": current_price.round(2),
        "n_price_rises_since_joining": n_price_rises,
        "months_since_last_price_rise": months_since_last_price_rise,
        "region": region,
        "primary_device": primary_device,
        "num_devices_registered": num_devices_registered,
        "household_size_est": household_size_est,
        "contract_type": contract_type,
        "months_to_contract_end": months_to_contract_end,
        "support_contacts_last_90d": support_contacts_last_90d,
        "billing_dispute_last_90d": billing_dispute_last_90d,
        "outage_reported_last_90d": outage_reported_last_90d,
        "nps_last_survey": nps_last_survey,
        "competitor_ad_exposure_score": competitor_ad_exposure_score.round(3),
    })
    return df


def generate_viewing_activity(subscribers: pd.DataFrame):
    """Monthly genre-level viewing hours for the last 3 months per account."""
    n = len(subscribers)
    rows = []
    # Each subscriber has an underlying genre affinity profile (Dirichlet)
    affinity = RNG.dirichlet(alpha=np.ones(len(GENRES)) * 0.7, size=n)
    base_engagement = RNG.gamma(shape=2.0, scale=9.0, size=n)  # total hrs/month baseline

    for month_offset in [2, 1, 0]:  # months ago
        monthly_noise = RNG.normal(1.0, 0.18, size=n).clip(0.3, 2.0)
        total_hours = (base_engagement * monthly_noise).clip(0, None)
        genre_hours = affinity * total_hours[:, None]
        month_df = pd.DataFrame(genre_hours, columns=[f"hrs_{g}" for g in GENRES])
        month_df.insert(0, "account_id", subscribers["account_id"].values)
        month_df.insert(1, "months_ago", month_offset)
        month_df["total_viewing_hours"] = total_hours.round(2)
        rows.append(month_df)

    activity = pd.concat(rows, ignore_index=True)
    for g in GENRES:
        activity[f"hrs_{g}"] = activity[f"hrs_{g}"].round(2)
    return activity, base_engagement


def label_churn_and_upsell(df: pd.DataFrame, base_engagement: np.ndarray):
    """Assign churn (next 60 days) and upsell-propensity labels using a
    logit model with realistic, literature-consistent drivers."""
    n = len(df)
    engagement_z = (base_engagement - base_engagement.mean()) / base_engagement.std()

    price_rise_recency_risk = np.where(df["months_since_last_price_rise"] <= 2, 1.0, 0.0)
    out_of_contract_risk = np.where(df["months_to_contract_end"] <= 0, 1.0, 0.0)

    churn_logit = (
        -2.6
        - 0.55 * engagement_z
        + 0.9 * price_rise_recency_risk
        + 0.5 * out_of_contract_risk
        + 0.35 * df["support_contacts_last_90d"].clip(0, 4)
        + 0.9 * df["billing_dispute_last_90d"]
        + 0.5 * df["outage_reported_last_90d"]
        - 0.09 * df["nps_last_survey"]
        + 0.7 * df["competitor_ad_exposure_score"]
        - 0.15 * np.log1p(df["tenure_months"])
        + 0.010 * df["current_monthly_price_gbp"]
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = RNG.binomial(1, churn_prob)

    # Upsell propensity: to a higher-tier bundle (e.g. Sky Sports / Cinema add-on)
    # Driven by high engagement, low current tier, and NOT already on top plan.
    already_top = (df["plan"] == "Ultimate TV").astype(int)
    upsell_logit = (
        -1.8
        + 0.75 * engagement_z
        - 1.2 * already_top
        + 0.35 * df["num_devices_registered"]
        + 0.05 * df["nps_last_survey"]
        - 0.4 * out_of_contract_risk * 0  # neutral factor, kept for realism/documentation
    )
    upsell_prob = 1 / (1 + np.exp(-upsell_logit))
    upsell_taken = RNG.binomial(1, upsell_prob)

    df = df.copy()
    df["churn_probability_true"] = churn_prob.round(4)  # ground truth, not used in training
    df["churned_next_60d"] = churned
    df["upsell_probability_true"] = upsell_prob.round(4)
    df["accepted_upsell_offer"] = upsell_taken
    return df


if __name__ == "__main__":
    subscribers = generate_subscribers()
    activity, base_engagement = generate_viewing_activity(subscribers)
    subscribers = label_churn_and_upsell(subscribers, base_engagement)

    subscribers.to_csv("data/subscribers.csv", index=False)
    activity.to_csv("data/viewing_activity.csv", index=False)

    print(f"subscribers.csv: {subscribers.shape}")
    print(f"viewing_activity.csv: {activity.shape}")
    print(f"Churn rate: {subscribers['churned_next_60d'].mean():.3%}")
    print(f"Upsell acceptance rate: {subscribers['accepted_upsell_offer'].mean():.3%}")
