"""
Unit tests for visualization module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.visualization import visualize_grid_search_results, export_best_strategies

@pytest.fixture
def mock_plt():
    with patch('trading_strategy.visualization.plt') as mock:
        yield mock

@pytest.fixture
def mock_sns():
    with patch('trading_strategy.visualization.sns') as mock:
        yield mock

def test_visualize_grid_search_results(mock_plt, mock_sns, tmp_path):
    results = pd.DataFrame({
        'sharpe_ratio': np.random.randn(10),
        'win_rate': np.random.rand(10),
        'total_return': np.random.randn(10),
        'max_drawdown': -np.random.rand(10),
        'sortino_ratio': np.random.randn(10),
        'calmar_ratio': np.random.randn(10),
        'n_trades': np.random.randint(1, 100, 10),
        'profit_factor': np.random.rand(10) * 2,
        'strategy_name': ['Strat'] * 10
    })
    
    # Mock mocks
    fig_mock = MagicMock()
    # Create an axes mock that handles [x, y] indexing
    axes_mock = MagicMock()
    def get_ax(key):
        return MagicMock()
    axes_mock.__getitem__.side_effect = get_ax
    
    mock_plt.subplots.return_value = (fig_mock, axes_mock)
    
    save_path = tmp_path / "test_plot.png"
    visualize_grid_search_results(results, save_path=str(save_path))
    
    # Check if subplots was called
    mock_plt.subplots.assert_called()
    # Check if savefig was called
    mock_plt.savefig.assert_called()

def test_export_best_strategies(tmp_path):
    results = pd.DataFrame({
        'sharpe_ratio': [1.0, 2.0, 1.5],
        'name': ['A', 'B', 'C'],
        'params': [{}, {}, {}]
    })
    
    save_path = tmp_path / "best.json"
    
    export_best_strategies(results, top_n=2, save_path=str(save_path))
    
    assert save_path.exists()
