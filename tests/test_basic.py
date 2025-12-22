-"""
Tests básicos para el módulo trading_strategy

Para ejecutar:
    pytest tests/test_basic.py -v
    
O todos los tests:
    pytest tests/ -v
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy import (
    calculate_returns_and_momentum,
    calculate_indicator_and_signals,
    backtest_strategy
)


# ========== FIXTURES ==========

@pytest.fixture
def sample_ohlcv_data():
    """
    Genera DataFrame OHLCV de prueba
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    
    # Generar precios con tendencia y volatilidad
    close_prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)
    
    df = pd.DataFrame({
        'Open': close_prices + np.random.randn(1000) * 0.2,
        'High': close_prices + np.abs(np.random.randn(1000) * 0.5),
        'Low': close_prices - np.abs(np.random.randn(1000) * 0.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    return df


# ========== TESTS DE INDICADORES ==========

def test_calculate_returns_and_momentum_basic(sample_ohlcv_data):
    """Test que calculate_returns_and_momentum calcula retornos básicos"""
    df = calculate_returns_and_momentum(
        sample_ohlcv_data, 
        compute_indicators=False
    )
    
    assert 'returns' in df.columns
    assert 'log_returns' in df.columns
    assert 'candle_rtn' in df.columns
    assert not df['returns'].isna().all()


def test_calculate_returns_and_momentum_with_indicators(sample_ohlcv_data):
    """Test que calculate_returns_and_momentum calcula indicadores"""
    df = calculate_returns_and_momentum(
        sample_ohlcv_data, 
        compute_indicators=True
    )
    
    # Verificar que algunos indicadores clave existen
    expected_indicators = ['rsi_14', 'macd', 'stoch_k', 'returns']
    
    for indicator in expected_indicators:
        assert indicator in df.columns, f"Indicador {indicator} no encontrado"


def test_calculate_indicator_rsi(sample_ohlcv_data):
    """Test que calculate_indicator_and_signals genera señales RSI"""
    df = sample_ohlcv_data.copy()
    df['returns'] = df['Close'].pct_change()
    
    result = calculate_indicator_and_signals(
        df, 
        'rsi', 
        {'period': 14, 'overbought': 70, 'oversold': 30}
    )
    
    assert 'signal' in result.columns
    assert result['signal'].isin([-1, 0, 1]).all()
    assert not result['signal'].isna().all()


# ========== TESTS DE BACKTESTING ==========

def test_backtest_strategy_returns_metrics(sample_ohlcv_data):
    """Test que backtest_strategy retorna métricas correctas"""
    df = sample_ohlcv_data.copy()
    df['returns'] = df['Close'].pct_change()
    
    metrics = backtest_strategy(
        df=df,
        indicator='rsi',
        params={'period': 14, 'overbought': 70, 'oversold': 30},
        position_type='long'
    )
    
    # Puede retornar None si no hay suficientes trades
    if metrics is not None:
        # Verificar que todas las métricas esperadas existen
        expected_metrics = [
            'sharpe_ratio', 'win_rate', 'profit_factor', 
            'max_drawdown', 'total_return', 'n_trades'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Métrica {metric} no encontrada"
        
        # Verificar tipos y rangos
        assert isinstance(metrics['n_trades'], int)
        assert metrics['n_trades'] >= 0
        assert 0 <= metrics['win_rate'] <= 1


def test_backtest_strategy_with_insufficient_trades(sample_ohlcv_data):
    """Test que backtest_strategy retorna None con pocos trades"""
    # Generar datos con muy pocos cambios
    df = sample_ohlcv_data.head(20).copy()
    df['returns'] = df['Close'].pct_change()
    
    metrics = backtest_strategy(
        df=df,
        indicator='rsi',
        params={'period': 14, 'overbought': 99, 'oversold': 1},  # Thresholds extremos
        position_type='long'
    )
    
    # Debe retornar None si hay menos de 10 trades
    assert metrics is None or metrics['n_trades'] >= 10


# ========== TESTS DE EDGE CASES ==========

def test_empty_dataframe():
    """Test manejo de DataFrame vacío"""
    df = pd.DataFrame()
    
    with pytest.raises((KeyError, ValueError, AttributeError)):
        calculate_returns_and_momentum(df, compute_indicators=False)


def test_missing_columns():
    """Test manejo de columnas faltantes"""
    df = pd.DataFrame({
        'Close': [100, 101, 102]
    })
    
    with pytest.raises((KeyError, ValueError, AttributeError)):
        calculate_returns_and_momentum(df, compute_indicators=False)


def test_nan_values_handling(sample_ohlcv_data):
    """Test que el sistema maneja NaN correctamente"""
    df = sample_ohlcv_data.copy()
    df.loc[df.index[0:10], 'Close'] = np.nan
    
    result = calculate_returns_and_momentum(df, compute_indicators=False)
    
    # Debe haber eliminado las filas con NaN en returns críticos
    assert result['returns'].isna().sum() == 0
    assert len(result) < len(df)


# ========== TESTS DE INTEGRACIÓN ==========

def test_full_pipeline(sample_ohlcv_data):
    """Test del pipeline completo: datos -> indicadores -> backtest"""
    # 1. Calcular retornos
    df = calculate_returns_and_momentum(
        sample_ohlcv_data, 
        compute_indicators=False
    )
    assert 'returns' in df.columns
    
    # 2. Ejecutar backtest
    metrics = backtest_strategy(
        df=df,
        indicator='rsi',
        params={'period': 14, 'overbought': 70, 'oversold': 30},
        position_type='both'
    )
    
    # 3. Verificar resultado
    if metrics is not None:
        assert 'sharpe_ratio' in metrics
        assert isinstance(metrics['n_trades'], int)


# ========== TESTS DE PERFORMANCE ==========

def test_performance_large_dataset():
    """Test de performance con dataset grande"""
    import time
    
    # Generar dataset grande
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=10000, freq='1h')
    close_prices = 100 + np.cumsum(np.random.randn(10000) * 0.5)
    
    df = pd.DataFrame({
        'Open': close_prices,
        'High': close_prices + 1,
        'Low': close_prices - 1,
        'Close': close_prices,
        'Volume': 1000
    }, index=dates)
    
    # Medir tiempo
    start = time.time()
    result = calculate_returns_and_momentum(df, compute_indicators=False)
    elapsed = time.time() - start
    
    # Debe completarse en menos de 5 segundos
    assert elapsed < 5.0, f"Cálculo demasiado lento: {elapsed:.2f}s"
    assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
