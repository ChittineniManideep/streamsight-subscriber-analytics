"""
train_churn_model.py
---------------------
Trains a subscriber churn model (XGBoost) on the engineered feature set.
This script is the "local equivalent" of the Vertex AI custom training job
defined in vertex_ai/pipeline.py — same feature contract, same model code,
so it can be validated locally before being containerised and submitted
as a Vertex AI CustomTrainingJob.
"""

import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, "src")
from features import build_churn_training_frame, build_viewing_features, CATEGORICAL_COLS, NUMERIC_COLS


def load_data():
    subscribers = pd.read_csv("data/subscribers.csv")
    activity = pd.read_csv("data/viewing_activity.csv")
    viewing_features = build_viewing_features(activity)
    return build_churn_training_frame(subscribers, viewing_features)


def build_pipeline():
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        scale_pos_weight=6.5,  # ~13% churn base rate -> upweight positive class
        random_state=42,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    df = load_data()
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    # Business-relevant cut: top-decile risk accounts (what a retention team
    # would actually action via targeted save offers)
    threshold = np.quantile(proba, 0.90)
    preds_top_decile = (proba >= threshold).astype(int)
    report = classification_report(y_test, preds_top_decile, output_dict=True)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    # capture-rate at top decile: what share of ALL churners we catch by
    # targeting only the highest-risk 10% of the base
    top_decile_idx = proba >= threshold
    capture_rate = y_test[top_decile_idx].sum() / y_test.sum()

    metrics = {
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "base_churn_rate": round(float(y.mean()), 4),
        "top_decile_threshold": round(float(threshold), 4),
        "top_decile_precision": round(float(report["1"]["precision"]), 4),
        "top_decile_churn_capture_rate": round(float(capture_rate), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(json.dumps(metrics, indent=2))

    with open("reports/churn_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import joblib
    joblib.dump(pipe, "models/churn_model.joblib")

    # Feature importance for the model card / stakeholder readout
    ohe = pipe.named_steps["preprocess"].named_transformers_["cat"]
    feature_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS)) + NUMERIC_COLS
    importances = pipe.named_steps["model"].feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    fi.to_csv("reports/churn_feature_importance.csv", index=False)
    print("\nTop drivers of churn risk:")
    print(fi.to_string(index=False))


if __name__ == "__main__":
    main()
