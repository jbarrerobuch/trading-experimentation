"""
Trading Strategy Framework
Módulo para desarrollo y optimización de estrategias de trading basadas en momentum
"""

from .data_loader import load_saved_data, fetch_ohlcv_data
from .indicators import (
    calculate_returns_and_momentum,
    calculate_indicator_and_signals,
    combine_indicator_signals,
    clear_indicator_cache
)
from .backtesting import backtest_strategy, get_strategy_trades
from .grid_search import strategy_grid_search
from .validation_metrics import (
    build_validation_context,
    compute_validation_metrics,
    backtest_with_validation,
    compute_stability_table,
    decision_rule,
    dedupe_runs,
)
from .batch_grid_search import (
    batch_grid_search,
    estimate_batch_requirements
)
from .walk_forward import walk_forward_optimization
from .visualization import visualize_grid_search_results, export_best_strategies
from .strategy_loader import (
    load_strategy_config,
    load_all_strategies,
    load_strategies_by_name,
    get_strategy_info
)

__version__ = "1.0.0"

__all__ = [
    # Data loading
    'load_saved_data',
    'fetch_ohlcv_data',
    
    # Indicators
    'calculate_returns_and_momentum',
    'calculate_indicator_and_signals',
    'combine_indicator_signals',
    'clear_indicator_cache',
    
    # Backtesting
    'backtest_strategy',
    'get_strategy_trades',
    
    # Validation (anti-overfitting)
    'build_validation_context',
    'compute_validation_metrics',
    'backtest_with_validation',
    'compute_stability_table',
    'decision_rule',
    'dedupe_runs',

    # Grid Search
    'strategy_grid_search',
    'batch_grid_search',
    'estimate_batch_requirements',
    'walk_forward_optimization',
    
    # Visualization
    'visualize_grid_search_results',
    'export_best_strategies',
    
    # Strategy Loading
    'load_strategy_config',
    'load_all_strategies',
    'load_strategies_by_name',
    'get_strategy_info',
]
