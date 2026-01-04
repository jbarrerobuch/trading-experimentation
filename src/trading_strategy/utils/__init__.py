"""
Utilidades del paquete trading_strategy
"""

from .helpers import (
    calculate_returns_summary,
    normalize_ohlcv_columns,
    resample_ohlcv,
    format_strategy_config,
    generate_run_name
)

from .stats import (
    calculate_trade_statistics,
    is_outlier,
    is_outlier_mod_zscore
)

__all__ = [
    'calculate_returns_summary',
    'normalize_ohlcv_columns',
    'resample_ohlcv',
    'format_strategy_config',
    'generate_run_name',
    'calculate_trade_statistics',
    'is_outlier',
    'is_outlier_mod_zscore'
]
