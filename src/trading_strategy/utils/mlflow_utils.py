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

from .paths import get_project_root, get_data_dir

def get_mlflow_db_path() -> Path:
    """Returns the absolute path to the mlflow.db file."""
    return get_project_root() / 'mlflow.db'

def setup_mlflow(experiment_name: str = 'default', disable_gpu_metrics: bool = True) -> bool:
    """
    Configures MLflow to use SQLite database backend.
    
    Args:
        experiment_name: Name of the experiment to set/create.
        disable_gpu_metrics: If True, hides GPU from MLflow to prevent GPU metrics logging.
        
    Returns:
        bool: True if setup was successful, False otherwise.
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️  MLflow not installed. Tracking disabled.")
        return False

    try:
        # Ocultar GPU si se solicita (antes de inicializar cualquier cosa de MLflow)
        if disable_gpu_metrics:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            # También para AMD/ROCm por si acaso
            os.environ["ROCR_VISIBLE_DEVICES"] = ""
            
        db_path = get_mlflow_db_path()
        tracking_uri = f"sqlite:///{db_path}"
        
        mlflow.set_tracking_uri(tracking_uri)  # pyright: ignore[reportPossiblyUnboundVariable]
        
        # Configure artifact location to be in data/mlruns
        artifact_path = get_data_dir() / 'mlruns'
        artifact_path.mkdir(parents=True, exist_ok=True)
        # Ensure URI format is correct for MLflow
        artifact_uri = artifact_path.as_uri()

        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name) # pyright: ignore[reportPossiblyUnboundVariable]
        
        if experiment is None:
            print(f"Creating new experiment '{experiment_name}' with artifacts at {artifact_path}")
            mlflow.create_experiment(experiment_name, artifact_location=artifact_uri) # pyright: ignore[reportPossiblyUnboundVariable]
        
        mlflow.set_experiment(experiment_name) # pyright: ignore[reportPossiblyUnboundVariable]
        
        # Intentar activar System Metrics nativos (MLflow 2.8+)
        try:
            mlflow.enable_system_metrics_logging() # pyright: ignore[reportPossiblyUnboundVariable]
        except (AttributeError, Exception):
            pass # Versión antigua de MLflow o error de permisos
        
        print(f"✓ MLflow configured using SQLite: {db_path}")
        return True
    except Exception as e:
        print(f"⚠️  Error configuring MLflow: {e}")
        return False

def parse_combo_params(params: dict) -> list:
    """
    Reconstructs indicator structure for combo strategies from flat MLflow params.
    
    Args:
        params: Dictionary of parameters from MLflow run.
        
    Returns:
        List of dictionaries defining the indicators.
    """
    indicators_combo = []
    
    # Determine how many indicators there are
    n_indicators = int(params.get('n_indicators', 0))
    
    for i in range(1, n_indicators + 1):
        ind_name = params.get(f'ind{i}_name')
        if not ind_name:
            continue
            
        ind_params = {}
        prefix = f'ind{i}_'
        
        for key, value in params.items():
            if key.startswith(prefix) and key != f'{prefix}name':
                param_name = key[len(prefix):]
                # Try to convert to number if possible
                try:
                    if '.' in str(value):
                        ind_params[param_name] = float(value)
                    else:
                        ind_params[param_name] = int(value)
                except (ValueError, TypeError):
                    ind_params[param_name] = value
        
        indicators_combo.append({
            'indicator': ind_name,
            'params': ind_params
        })
        
    return indicators_combo

def parse_single_params(params: dict) -> dict:
    """
    Cleans and converts parameters for single strategies.
    
    Args:
        params: Dictionary of parameters from MLflow run.
        
    Returns:
        Dictionary of cleaned parameters.
    """
    clean_params = {}
    exclude_keys = {
        'strategy_name', 'strategy_type', 'indicator', 'position_type', 
        'train_split', 'ticker', 'timeframe', 'git_commit', 'session_id'
    }
    
    for k, v in params.items():
        if k not in exclude_keys and not k.startswith('ind'):
            try:
                if '.' in str(v):
                    clean_params[k] = float(v)
                else:
                    clean_params[k] = int(v)
            except (ValueError, TypeError):
                clean_params[k] = v
                
    return clean_params