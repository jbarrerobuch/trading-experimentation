"""
Unit tests for walk_forward module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.walk_forward import walk_forward_optimization

@pytest.fixture
def sample_df():
    dates = pd.date_range('2023-01-01', periods=100, freq='1d')
    df = pd.DataFrame({
        'Open': 100, 'High': 110, 'Low': 90, 'Close': 105, 'Volume': 1000, 'returns': 0.01
    }, index=dates)
    return df

@pytest.fixture
def mock_grid_search():
    with patch('trading_strategy.walk_forward.strategy_grid_search') as mock:
        yield mock

def test_walk_forward_optimization_basic(sample_df, mock_grid_search):
    # Mock grid search returning one result
    mock_grid_search.return_value = pd.DataFrame([{
        'sharpe_ratio': 1.0,
        'params': {'period': 14}, # Params representation depends on impl, usually string or dict
        'total_return': 0.1,
        # need enough info so walk_forward can select best
        'config_str': "{'period': 14}",
        'strategy_name': 'Test'
    }])
    
    # We need to ensure logic inside walk_forward uses the mock correctly.
    # walk_forward splits data, calls grid_search on training set.
    
    configs = [{
        'name': 'Test',
        'type': 'single',
        'indicator': 'rsi',
        'params_grid': {'period': [14]}
    }]
    
    # Run small WF
    # window=50, step=25. 100 rows.
    # 1. Train 0-50, Test 50-75
    # 2. Train 25-75, Test 75-100
    results = walk_forward_optimization(
        df=sample_df,
        strategy_configs=configs,
        window_size=50,
        step_size=25,
        metric='sharpe_ratio'
    )
    
    assert results is not None
    # We expect some dataframe with OOS results
    # Since we mocked grid search to return good metrics, logic should proceed.
    
    assert mock_grid_search.call_count >= 1

