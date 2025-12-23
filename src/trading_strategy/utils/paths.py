"""
Path utilities for the Trading Strategy Framework.
Handles directory resolution robustly.
"""
import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # This file is in src/trading_strategy/utils/paths.py
    # We need to go up 4 levels to reach 'main'
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent.parent

def get_data_dir() -> Path:
    """Returns the absolute path to the data directory."""
    return get_project_root() / 'data'

def get_market_data_dir() -> Path:
    """Returns the absolute path to the market data directory."""
    return get_data_dir() / 'market'

def get_strategies_dir() -> Path:
    """Returns the absolute path to the strategies directory."""
    return get_project_root() / 'strategies'

def get_mlruns_dir() -> Path:
    """Returns the absolute path to the mlruns directory."""
    return get_project_root() / 'mlruns'
