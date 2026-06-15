"""
Unit tests for portfolio module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.portfolio import PortfolioManager
from trading_strategy.constants import COL_CLOSE, COL_SIGNAL

@pytest.fixture
def market_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        COL_CLOSE: np.array([100.0] * 100), # Constant price to simplify, or variable
        'Volume': 1000
    }, index=dates)
    # Add varying prices to get non-zero returns
    df[COL_CLOSE] = 100 * (1 + 0.01 * np.random.randn(100)).cumsum()
    df['returns'] = df[COL_CLOSE].pct_change()
    return df

@pytest.fixture
def portfolio_manager(market_data):
    return PortfolioManager(market_data)

def test_init_portfolio_manager(market_data):
    pm = PortfolioManager(market_data)
    assert not pm.market_data.empty
    assert 'returns' in pm.market_data.columns

def test_add_strategy_and_returns(portfolio_manager):
    # Mock calculate_indicator_and_signals by patching? 
    # Or rely on simplified logic. 
    # PortfolioManager calls calculate_indicator_and_signals if indicator provided.
    # If indicator is None, returns 0.
    
    # Let's use real simple params or mock internal call
    with pytest.MonkeyPatch.context() as m:
        # Mock calculation to return a DF with specific signal
        def mock_calc(df, indicator, params, inplace):
            # Generate alternating signals
            df[COL_SIGNAL] = np.tile([1, -1, 0, 0], len(df) // 4 + 1)[:len(df)]
            return df
        
        m.setattr("trading_strategy.portfolio.calculate_indicator_and_signals", mock_calc)
        
        portfolio_manager.add_strategy(
            name="TestStrat",
            indicator="mock_ind",
            params={'a': 1}
        )
        
        assert "TestStrat" in portfolio_manager.strategies
        assert "TestStrat" in portfolio_manager.strategy_returns.columns
        # Since logic applies signal to returns, and we have non-zero returns in fixture,
        # we expect non-zero strategy returns (minus costs)
        assert portfolio_manager.strategy_returns["TestStrat"].sum() != 0

def test_simulate_portfolio(portfolio_manager):
    # Manually populate strategy returns to avoid dependency on calc logic
    portfolio_manager.strategy_returns['S1'] = pd.Series(0.01, index=portfolio_manager.market_data.index)
    portfolio_manager.strategy_returns['S2'] = pd.Series(-0.005, index=portfolio_manager.market_data.index)
    
    # Equal weight: (0.01 - 0.005) / 2 = 0.0025
    combined = portfolio_manager.simulate_portfolio(['S1', 'S2'])
    assert np.isclose(combined.iloc[0], 0.0025)
    
    # Weighted: S1=0.8, S2=0.2 -> 0.01*0.8 + (-0.005)*0.2 = 0.008 - 0.001 = 0.007
    combined_w = portfolio_manager.simulate_portfolio(['S1', 'S2'], weights=[0.8, 0.2])
    assert np.isclose(combined_w.iloc[0], 0.007)

def test_calculate_portfolio_metrics(portfolio_manager):
    # Series of 1% returns for 100 periods
    returns = pd.Series([0.01] * 100)
    metrics = portfolio_manager.calculate_portfolio_metrics(returns)
    
    assert metrics['total_return'] > 0
    assert metrics['sharpe_ratio'] > -100 # just check it calculates
    assert isinstance(metrics['max_drawdown'], float)

def test_find_best_combinations(portfolio_manager):
    portfolio_manager.strategy_returns['A'] = pd.Series(np.random.randn(100)*0.01 + 0.001, index=portfolio_manager.market_data.index)
    portfolio_manager.strategy_returns['B'] = pd.Series(np.random.randn(100)*0.01 + 0.001, index=portfolio_manager.market_data.index)
    portfolio_manager.strategy_returns['C'] = pd.Series(np.random.randn(100)*0.01 - 0.001, index=portfolio_manager.market_data.index)
    
    portfolio_manager.strategies = {'A': {}, 'B': {}, 'C': {}}
    
    results = portfolio_manager.find_best_combinations(min_k=2, max_k=2, top_n=5)
    
    assert not results.empty
    assert 'strategies' in results.columns
    assert len(results) <= 3 # 3 pairs: AB, AC, BC
