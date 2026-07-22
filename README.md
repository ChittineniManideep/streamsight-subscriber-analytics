# StreamSight — Subscriber Churn, Propensity & Content Affinity on GCP

A subscription-media analytics pipeline covering the three problems a
subscriber analytics team is typically asked to own: **churn risk**,
**upsell propensity**, and **content affinity segmentation** — built on a
**BigQuery + Vertex AI** architecture.

> Built as a portfolio project applying my existing analytics toolkit
> (customer segmentation, demand forecasting, behavioural modelling — from
> banking/commercial data at Tiger Analytics) to a subscription-media
> domain, and to demonstrate hands-on GCP depth (BigQuery feature
> engineering + a real, compiled Vertex AI Pipelines training/deployment
> DAG) beyond a one-line skills-list mention.

## Business problems addressed

| Problem | Question answered | Who acts on it |
|---|---|---|
| **Churn prediction** | Which accounts are likely to cancel in the next 60 days, and why? | Retention team — targeted save offers |
| **Upsell propensity** | Which accounts are likely to accept an upgrade offer (e.g. add Sky Sports/Cinema)? | Marketing — campaign targeting |
| **Content affinity** | What viewing personas exist in the base (sport-led, movie-led, kids, etc.)? | Content strategy — commissioning, scheduling, creative targeting |

## Architecture

```
GCS (raw CSV)                                                                 
   │                                                                          
   ▼                                                                          
BigQuery  ── streamsight_raw.subscribers, streamsight_raw.viewing_activity    
   │                                                                          
   ▼  (sql/02_feature_engineering.sql)                                        
BigQuery views ── viewing_features, churn_training_view, upsell_training_view 
   │                                                                          
   ▼  (vertex_ai/pipeline.py — compiled Vertex AI Pipelines DAG)              
Vertex AI Pipelines:                                                          
  extract_features_from_bigquery                                             
        → train_churn_model_component (XGBoost)                              
        → check_quality_gate (min ROC-AUC)                                   
        → register_and_deploy_model  (Model Registry → Endpoint)             
```

Data residency is set to `europe-west2` (London) throughout, matching a
UK subscriber base's likely regulatory requirement.

## Results (on the synthetic 50k-account dataset)

**Churn model** (XGBoost, class-weighted for a 13.4% base churn rate):
- ROC-AUC: **0.70** · Average precision: **0.27**
- Targeting the top-decile risk accounts catches **24%** of all churners at **32% precision** (~2.4x the base rate) — a realistic, actionable list size for a retention campaign.
- Top drivers: recency of last price rise, viewing engagement level, NPS, support contacts, billing disputes, outages, and out-of-contract status — consistent with published subscription-churn research.

**Upsell propensity model** (Gradient Boosting):
- ROC-AUC: **0.71** · Average precision: **0.62** (vs. 38% base acceptance rate)

**Content affinity segmentation** (k-means on genre-viewing shares):
- 7 clean, near-single-genre-dominant personas selected by silhouette score (sport, movies, drama/entertainment, kids, documentary, news, reality), each ~14% of the base.

Full walkthrough with charts: [`notebooks/01_eda_and_results.ipynb`](notebooks/01_eda_and_results.ipynb) ([HTML export](notebooks/01_eda_and_results.html)).

## Repository structure

```
data/                   synthetic data generator + output CSVs
sql/                    BigQuery DDL + feature engineering views
src/                    feature engineering + model training (local/pandas)
notebooks/              EDA, model results, business narrative
vertex_ai/              Vertex AI Pipelines (KFP v2) training/deployment DAG
models/                 trained model artifacts (.joblib)
reports/                metrics (JSON), feature importance, figures
```

## Why synthetic data, and why it's still a meaningful signal

No real subscriber data is used or available for this project. The
generator (`data/generate_subscriber_data.py`) builds features and a
churn/upsell label from a logistic model whose coefficients are set to
match known, published churn drivers for subscription and telco
services (price-rise recency, engagement decline, support friction,
contract status) rather than being arbitrary. The resulting ROC-AUC
(~0.70) is deliberately realistic — subscription churn models in
production commonly sit in the 0.65–0.75 range; a suspiciously high
number here would be a sign of label leakage, not of a better model.

## Running it

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python data/generate_subscriber_data.py

# 2. Train the models
python src/train_churn_model.py
python src/train_propensity_model.py
python src/content_affinity.py

# 3. Reproduce the notebook
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_eda_and_results.ipynb
```

### Deploying the BigQuery + Vertex AI layer against a real GCP project

```bash
# BigQuery: load raw data and build feature views
bq mk --dataset --location=europe-west2 streamsight_raw
# (upload data/*.csv to GCS first, then bq load — see sql/01_load_and_schema.sql)
bq query --use_legacy_sql=false < sql/02_feature_engineering.sql

# Vertex AI Pipelines: compile and submit the training/deployment DAG
export GCP_PROJECT_ID=<your-project>
python -c "from vertex_ai.pipeline import compile_pipeline, submit_pipeline_job; compile_pipeline(); submit_pipeline_job()"
```

`vertex_ai/pipeline.py` compiles to a valid Vertex AI Pipelines spec
(`vertex_ai/churn_pipeline.json`, included in this repo) using the real
`kfp` and `google-cloud-aiplatform` SDKs — it isn't run against a live
project here (no GCP credentials in this environment), but the DAG,
component contracts, and SDK calls are exactly what would execute against
one, including the BigQuery read, the custom training step, an evaluation
gate before promotion, and a Vertex AI Endpoint deployment.

## Notes on scope

This is a portfolio project built to demonstrate the analytical and GCP
tooling relevant to subscriber analytics roles — it is not affiliated
with, and does not use any data from, any real subscription or media
company. Plan names and regions are illustrative.
