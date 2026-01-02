import mlflow
import os
import sys
from pathlib import Path

# Setup path to find src if needed (though we just use mlflow here)
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
db_uri = f"sqlite:///{project_root}/mlflow.db"

print(f"Connecting to MLflow DB: {db_uri}")
mlflow.set_tracking_uri(db_uri)

try:
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments.")
    
    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        print(f"  Artifact Location: {exp.artifact_location}")
        print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
        
        # Search runs in this experiment
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Total Runs: {len(runs)}")
        
        if not runs.empty:
            print("  Sample Run Data (first 1):")
            print(runs.iloc[0].to_dict())
        else:
            print("  No runs found in this experiment.")

except Exception as e:
    print(f"Error accessing MLflow: {e}")
