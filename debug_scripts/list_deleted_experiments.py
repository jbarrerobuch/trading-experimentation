
import mlflow
from mlflow.tracking import MlflowClient
import os

# Set tracking URI
db_path = os.path.abspath("mlflow.db")
tracking_uri = f"sqlite:///{db_path}"
mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)

print(f"Found {len(experiments)} deleted experiments:")
for exp in experiments:
    print(f"ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle: {exp.lifecycle_stage}")
