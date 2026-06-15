"""
Unit tests for utils modules (stats, helpers, paths)
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.utils.stats import is_outlier, calculate_trade_statistics
from trading_strategy.utils.paths import get_project_root

def test_is_outlier():
    data = pd.Series([10, 10, 11, 10, 1000]) # 1000 is obvious outlier
    outliers = is_outlier(data)
    assert outliers.iloc[-1] == True
    assert outliers.iloc[0] == False

def test_calculate_trade_statistics_empty():
    df = pd.DataFrame()
    stats = calculate_trade_statistics(df)
    assert stats['total_trades'] == 0
    assert stats['total_return'] == 0.0

def test_calculate_trade_statistics_basic():
    df = pd.DataFrame({
        'pnl_pct': [0.1, -0.05, 0.2, -0.1],
        'pnl_usd': [100, -50, 200, -100],
        'net_pnl_usd': [90, -60, 190, -110], # Assuming some costs
        'entry_time': pd.date_range('2023-01-01', periods=4),
        'exit_time': pd.date_range('2023-01-02', periods=4)
    })
    
    stats = calculate_trade_statistics(df)
    
    assert stats['total_trades'] == 4
    assert stats['win_rate'] == 0.5
    assert stats['profit_factor'] > 0
    assert 'max_drawdown' in stats

def test_get_project_root():
    root = get_project_root()
    assert isinstance(root, (str, Path))
    if isinstance(root, Path):
        root = str(root)
    assert os.path.isdir(root)
    # Check if 'src' or 'requirements.txt' or similar exists in root
    # Adjust check based on actual structure
    assert os.path.exists(os.path.join(root, 'requirements.txt')) or \
           os.path.exists(os.path.join(root, 'src'))
