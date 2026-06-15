"""
Unit tests for data_loader module
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.data_loader import fetch_ohlcv_data, load_saved_data
from trading_strategy.constants import COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME

@pytest.fixture
def mock_ccxt_binance():
    with patch('trading_strategy.data_loader.ccxt.binance') as mock:
        yield mock

@pytest.fixture
def mock_market_dir(tmp_path):
    with patch('trading_strategy.data_loader.get_market_data_dir', return_value=tmp_path) as mock:
        yield tmp_path

def test_fetch_ohlcv_data_success(mock_ccxt_binance, mock_market_dir):
    # Mock exchange instance
    exchange = MagicMock()
    mock_ccxt_binance.return_value = exchange
    exchange.parse8601.return_value = 1000000
    exchange.milliseconds.return_value = 2000000
    
    # Mock OHLCV data
    # [timestamp, open, high, low, close, volume]
    mock_ohlcv = [
        [1000000, 100, 110, 90, 105, 1000],
        [1000060, 105, 115, 95, 110, 1200]
    ]
    # First call returns data, second call returns numeric timestamp indicating end or empty
    # The loop condition is 'if not ohlcv: break' OR 'if ohlcv[-1][0] >= exchange.milliseconds(): break'
    # We'll make it return one batch then trigger end
    exchange.fetch_ohlcv.side_effect = [mock_ohlcv, []]
    
    result = fetch_ohlcv_data(
        ticker='BTC/USDT',
        timeframes=['1h'],
        start_date='2017-08-17',
        save_data=True
    )
    
    assert '1h' in result
    df = result['1h']
    assert len(df) == 2
    assert COL_CLOSE in df.columns
    assert df.index.name == 'timestamp' # or whatever COL_TIMESTAMP maps to, usually index
    
    # Check if file was saved
    files = list(mock_market_dir.glob('*.parquet'))
    assert len(files) == 1
    assert 'btcusdt_1h_' in files[0].name

def test_fetch_ohlcv_data_no_data(mock_ccxt_binance, mock_market_dir):
    exchange = MagicMock()
    mock_ccxt_binance.return_value = exchange
    exchange.parse8601.return_value = 1000000
    exchange.fetch_ohlcv.return_value = []
    
    result = fetch_ohlcv_data('BTC/USDT', ['1h'], '2023-01-01')
    # The code constructs a DataFrame even if data is empty, as long as the loop ran once.
    # Since the loop runs over timeframes, it will create an empty DF for '1h'.
    assert '1h' in result
    assert result['1h'].empty

def test_load_saved_data_success(mock_market_dir):
    # Create dummy parquet file
    df = pd.DataFrame({
        COL_OPEN: [100.0],
        COL_HIGH: [110.0],
        COL_LOW: [90.0],
        COL_CLOSE: [105.0],
        COL_VOLUME: [1000.0]
    }, index=pd.to_datetime(['2023-01-01']))
    
    ticker_clean = 'btcusdt'
    tf = '1h'
    filename = f'{ticker_clean}_{tf}_20230101_20230101.parquet'
    df.to_parquet(mock_market_dir / filename)
    
    result = load_saved_data('BTC/USDT', ['1h'])
    
    assert '1h' in result
    assert len(result['1h']) == 1
    assert result['1h'].iloc[0][COL_CLOSE] == 105.0

def test_load_saved_data_no_files(mock_market_dir):
    result = load_saved_data('BTC/USDT', ['1h'])
    assert result == {}
