"""
content_affinity.py
---------------------
Segments subscribers into content-affinity personas using their genre
viewing-share vector (sport / movies / drama / kids / documentary / news /
reality). This is the "content affinity" capability referenced in the JD —
distinct from churn/propensity because it's unsupervised: no label, just
behavioural grouping used for content recommendation, scheduling, and
targeted marketing creative (e.g. sport-led save offers for sport-affine
accounts about to churn).
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")
from features import build_viewing_features, GENRES

AFFINITY_COLS = [f"affinity_{g}" for g in GENRES]


def load_affinity_matrix():
    activity = pd.read_csv("data/viewing_activity.csv")
    viewing_features = build_viewing_features(activity)
    affinity = viewing_features[["account_id"] + AFFINITY_COLS].dropna()
    return affinity


def fit_segments(affinity: pd.DataFrame, k_range=range(3, 9)):
    X = affinity[AFFINITY_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        # sample for silhouette speed on 50k rows
        sample_idx = np.random.default_rng(42).choice(len(X_scaled), size=5000, replace=False)
        score = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
        scores[k] = round(float(score), 4)

    best_k = max(scores, key=scores.get)
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    affinity = affinity.copy()
    affinity["segment"] = km_final.fit_predict(X_scaled)

    return affinity, km_final, scaler, scores, best_k


def profile_segments(affinity: pd.DataFrame):
    profile = affinity.groupby("segment")[AFFINITY_COLS].mean().round(3)
    profile["n_accounts"] = affinity.groupby("segment").size()
    # Label each segment by its top genre for readability
    profile["dominant_genre"] = profile[AFFINITY_COLS].idxmax(axis=1).str.replace("affinity_", "")
    return profile


def main():
    affinity = load_affinity_matrix()
    segmented, model, scaler, scores, best_k = fit_segments(affinity)
    profile = profile_segments(segmented)

    print(f"Silhouette scores by k: {scores}")
    print(f"Selected k = {best_k}\n")
    print(profile.to_string())

    joblib.dump({"model": model, "scaler": scaler, "affinity_cols": AFFINITY_COLS},
                "models/content_affinity_segments.joblib")
    profile.to_csv("reports/content_affinity_segments.csv")
    segmented[["account_id", "segment"]].to_csv("reports/account_segment_assignments.csv", index=False)

    with open("reports/content_affinity_metrics.json", "w") as f:
        json.dump({"k_selected": best_k, "silhouette_by_k": scores}, f, indent=2)


if __name__ == "__main__":
    main()
