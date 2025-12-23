"""
Constants for the Trading Strategy Framework.
Centralizes column names and other constant values to avoid magic strings.
"""

# OHLCV Columns
COL_TIMESTAMP = 'timestamp'
COL_OPEN = 'Open'
COL_HIGH = 'High'
COL_LOW = 'Low'
COL_CLOSE = 'Close'
COL_VOLUME = 'Volume'

# Calculated Columns
COL_RETURNS = 'returns'
COL_LOG_RETURNS = 'log_returns'
COL_CANDLE_RTN = 'candle_rtn'
COL_FUTURE_RET = 'future_ret' # Base name, usually appended with +N
COL_SIGNAL = 'signal'

# Default Values
DEFAULT_TIMEFRAMES = ['1h', '4h', '1d']
DEFAULT_TICKER = 'BTCUSDT'
