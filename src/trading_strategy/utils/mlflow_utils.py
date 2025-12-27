"""
MLflow utilities for experiment tracking.
Handles configuration, database connection, and run naming.
"""

import os
import sys
from pathlib import Path

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from .paths import get_project_root

def get_mlflow_db_path() -> Path:
    """Returns the absolute path to the mlflow.db file."""
    return get_project_root() / 'mlflow.db'

def setup_mlflow(experiment_name: str = 'default') -> bool:
    """
    Configures MLflow to use SQLite database backend.
    
    Args:
        experiment_name: Name of the experiment to set/create.
        
    Returns:
        bool: True if setup was successful, False otherwise.
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️  MLflow not installed. Tracking disabled.")
        return False

    try:
        db_path = get_mlflow_db_path()
        tracking_uri = f"sqlite:///{db_path}"
        
        mlflow.set_tracking_uri(tracking_uri)  # pyright: ignore[reportPossiblyUnboundVariable]
        mlflow.set_experiment(experiment_name) # pyright: ignore[reportPossiblyUnboundVariable]
        
        print(f"✓ MLflow configured using SQLite: {db_path}")
        return True
    except Exception as e:
        print(f"⚠️  Error configuring MLflow: {e}")
        return False
