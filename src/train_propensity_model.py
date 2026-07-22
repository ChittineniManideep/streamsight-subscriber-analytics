"""
train_propensity_model.py
---------------------------
Trains an upsell/upgrade propensity model — predicts which subscribers are
likely to accept an offer to upgrade to a higher-tier bundle (e.g. adding
Sky Sports or Sky Cinema). Used to prioritise which accounts a marketing
campaign should target, and to avoid wasting offers on accounts unlikely
to convert or already at the top tier.
"""

import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
from features import build_upsell_training_frame, build_viewing_features, CATEGORICAL_COLS, NUMERIC_COLS


def load_data():
    subscribers = pd.read_csv("data/subscribers.csv")
    activity = pd.read_csv("data/viewing_activity.csv")
    viewing_features = build_viewing_features(activity)
    return build_upsell_training_frame(subscribers, viewing_features)


def main():
    df = load_data()
    # Exclude accounts already on the top tier — no upsell offer applies
    df = df[df["plan"] != "Ultimate TV"].copy()

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )
    model = GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    metrics = {
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "base_acceptance_rate": round(float(y.mean()), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_eligible_accounts": len(df),
    }
    print(json.dumps(metrics, indent=2))

    with open("reports/propensity_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, "models/propensity_model.joblib")


if __name__ == "__main__":
    main()
