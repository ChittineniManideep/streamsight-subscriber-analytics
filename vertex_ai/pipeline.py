"""
pipeline.py
------------
Vertex AI Pipelines (KFP v2) definition for the churn model. This compiles
to a pipeline spec (JSON) that can be submitted to Vertex AI Pipelines,
orchestrating: BigQuery feature extraction -> custom training on Vertex AI
-> model evaluation gate -> conditional upload to Vertex AI Model Registry
-> deployment to a Vertex AI Endpoint.

Requires: pip install google-cloud-aiplatform kfp google-cloud-bigquery
This script is written to be genuinely submittable against a real GCP
project (PROJECT_ID / BUCKET set via env vars) — it is not run in this
sandbox, which has no GCP credentials, but the DAG, component contracts,
and SDK calls below are what would actually execute on Vertex AI.
"""

import os

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, component

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "streamsight-analytics")
REGION = os.environ.get("GCP_REGION", "europe-west2")
PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT", f"gs://{PROJECT_ID}-pipelines/streamsight")


@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-bigquery==3.25.0", "pandas==2.2.2", "pyarrow==16.1.0"],
)
def extract_features_from_bigquery(
    project: str,
    bq_view: str,
    output_dataset: Output[Dataset],
):
    """Reads the churn_training_view built in sql/02_feature_engineering.sql
    and materialises it as a training dataset artifact."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{bq_view}`"
    df = client.query(query).to_dataframe()
    df.to_parquet(output_dataset.path + ".parquet")


@component(
    base_image="python:3.11",
    packages_to_install=["scikit-learn==1.5.0", "xgboost==2.1.0", "pandas==2.2.2", "joblib==1.4.2"],
)
def train_churn_model_component(
    input_dataset: Input[Dataset],
    model_out: Output[Model],
    metrics_out: Output[Metrics],
):
    """Custom training step — same feature contract as src/train_churn_model.py,
    packaged to run as a Vertex AI CustomTrainingJob step inside the pipeline."""
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    import xgboost as xgb

    df = pd.read_parquet(input_dataset.path + ".parquet")
    categorical = ["plan", "region", "primary_device", "contract_type"]
    numeric = [c for c in df.columns if c not in categorical + ["account_id", "label"]]

    X, y = df[categorical + numeric], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ])
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
        scale_pos_weight=6.5, random_state=42,
    )
    pipe = Pipeline([("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    metrics_out.log_metric("roc_auc", float(roc_auc_score(y_test, proba)))
    metrics_out.log_metric("average_precision", float(average_precision_score(y_test, proba)))

    joblib.dump(pipe, model_out.path + ".joblib")


@component(base_image="python:3.11")
def check_quality_gate(metrics_in: Input[Metrics], min_auc: float) -> bool:
    """Blocks promotion to the Model Registry if the new model regresses
    below the minimum acceptable ROC-AUC threshold."""
    auc = metrics_in.metadata.get("roc_auc", 0.0)
    print(f"Trained model ROC-AUC: {auc}, gate threshold: {min_auc}")
    return auc >= min_auc


@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform==1.60.0"],
)
def register_and_deploy_model(
    project: str,
    region: str,
    model: Input[Model],
    display_name: str,
    serving_container_image_uri: str = (
        "europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.2-1:latest"
    ),
):
    """Registers the trained model to Vertex AI Model Registry and deploys
    it to an endpoint for real-time scoring. Runs only when the quality
    gate passes, so a regressed model never reaches production."""
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=region)

    uploaded = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model.path,
        serving_container_image_uri=serving_container_image_uri,
    )
    endpoint = uploaded.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
    )
    print(f"Deployed to endpoint: {endpoint.resource_name}")


@dsl.pipeline(
    name="streamsight-churn-training-pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def churn_training_pipeline(
    project: str = PROJECT_ID,
    bq_view: str = "streamsight_features.churn_training_view",
    min_auc: float = 0.65,
):
    extract_task = extract_features_from_bigquery(project=project, bq_view=bq_view)

    train_task = train_churn_model_component(input_dataset=extract_task.outputs["output_dataset"])

    gate_task = check_quality_gate(metrics_in=train_task.outputs["metrics_out"], min_auc=min_auc)

    with dsl.If(gate_task.output == True):  # noqa: E712
        register_and_deploy_model(
            project=project,
            region=REGION,
            model=train_task.outputs["model_out"],
            display_name="streamsight-churn-model",
        )


def compile_pipeline(output_path: str = "vertex_ai/churn_pipeline.json"):
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=churn_training_pipeline,
        package_path=output_path,
    )
    print(f"Compiled pipeline spec to {output_path}")


def submit_pipeline_job():
    """Submits the compiled pipeline to Vertex AI Pipelines. Requires
    `gcloud auth application-default login` and a configured project."""
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

    job = aiplatform.PipelineJob(
        display_name="streamsight-churn-training",
        template_path="vertex_ai/churn_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={"project": PROJECT_ID, "min_auc": 0.65},
    )
    job.submit()
    print(f"Submitted pipeline job: {job.resource_name}")


if __name__ == "__main__":
    compile_pipeline()
