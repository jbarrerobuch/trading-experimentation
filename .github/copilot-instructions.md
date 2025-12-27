# GammaNeutral Copilot Instructions

## Architecture & entry points
- The reusable trading strategy framework lives in `src/trading_strategy/`; its `__init__.py` exports the data loading, indicator, backtest, grid search, batch/grid helpers, visualization, and strategy loader helpers that other scripts call.
- `examples/basic_usage.py` shows the canonical pipeline: load parquet data from `data/market`, compute returns without recomputing all indicators, load YAML configs, run `strategy_grid_search`, and then visualize/export artifacts in `data/img` and `data/stats` for manual inspection.
- The `strategies/` folder holds YAML blueprints (see `strategies/README.md`) for both individual and combo indicators. Updating or adding strategies goes through `src/trading_strategy/strategy_loader.py`, which validates required keys (`name`, `indicator`/`indicators`, `params_grid`).

## Core data & experiment flow
- Data is persisted as Parquet under `data/market` (downloaded via `fetch_ohlcv_data()` and consumed via `load_saved_data()`). Keep the ticker naming consistent (lowercase, punctuation stripped) so `strategy_loader` finds files via `*_{tf}_*.parquet` patterns.
- `calculate_returns_and_momentum()` must be called before backtests/grid searches; it appends `returns`, `future_ret`, and indicator columns so `backtest_strategy()` can vectorize signals. If you only need returns, call that function with `compute_indicators=False` (used in the example to speed up grid search prep).
- Backtests (`backtest_strategy()`) return 16 metrics plus stats per combo (see `src/trading_strategy/backtesting.py`), so rely on those fields when writing visualizations or exports instead of recomputing metrics.

## Grid search & MLflow conventions
- Most automation happens through `strategy_grid_search()` and the batch-friendly `batch_grid_search()` (see `src/trading_strategy/batch_grid_search.py`); the latter splits configs/experiments into checkpoints and logs timing metadata `_batch_id`/`_session_id` to keep the runs resumable.
- When `use_mlflow=True`, the grid search code sets `mlflow.db` (SQLite) under the repo root and tags every run with `ticker`, `timeframe`, `strategy_type`, and a timestamp `session_id`. Keep `MLFLOW_TRACKING_URI` pointing at `sqlite:///mlflow.db` when you run `mlflow ui` (Quickstart commands and the README reiterate this).
- Logging is tolerant of missing MLflow; you can set `use_mlflow=False` to skip the dependency if the environment lacks it, but keep the `mlflow` package in `requirements.txt` for CI/workstations.

## Workflows & commands to know
- Install deps via `pip install -r requirements.txt`, then `cd examples` and run `python basic_usage.py` to exercise the full pipeline (data loading, grid search, visualization, export).
- To inspect experiments, start `mlflow ui --backend-store-uri sqlite:///mlflow.db` from the repo root and filter by tags such as `tags.ticker = 'BTCUSDT'` and `tags.timeframe = '1h'`.
- Tests live under `tests/test_basic.py`; run `pytest tests/ -v` (or targeting the file) to ensure indicator/backtest helpers behave, and reinstall fixtures if you change how returns or metrics are computed.

## Patterns & data conventions
- Strategy configs follow the `name/type/indicator/params_grid/position_type` structure (combo configs also describe `combination_methods` and a list of indicator blocks). Copy the exemplar YAMLs (`macd_optimization.yaml`, `rsi_macd_combo.yaml`, etc.) when adding new strategies, and rely on `load_strategies_by_name()` when you want a subset.
- Visual output should pass through `visualize_grid_search_results()` (six-panel dashboard) and `export_best_strategies()` from `src/trading_strategy/visualization.py` so that downstream tooling always finds files under `data/img` and `data/stats` with predictable naming.
- Use `compute_indicators=False` to pre-filter datasets when you only need returns for grid search and call `calculate_indicator_and_signals()` or `combine_indicator_signals()` within `backtest_strategy()` for more complex combos; these functions cache intermediate results, so call `clear_indicator_cache()` before a new ticker/timeframe.

## Known integrations & external expectations
- `ccxt` powers live data downloads (via `fetch_ohlcv_data()`), so expect rate-limits and wrap calls in retries or set `enableRateLimit` as shown in `data_loader.py` when you extend data pipelines.
- Indicators rely on `pandas-ta`, so keep that dependency pinned; if you need a custom indicator, register it in `indicators.py` and ensure it returns `signal` columns compatible with the binary/ternary logic in `backtest_strategy()`.
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `mlflow`, and `dateutil` appear in `requirements.txt`; prefer transverse stack in this repo instead of introducing unrelated heavy deps unless specifically required.

## Testing & validation cues
- `tests/test_basic.py` exercises the expected `calculate_returns_and_momentum()` column set, ensures RSI signal outputs stay within [-1,1], and verifies `backtest_strategy()` gracefully returns `None` when trades are sparse.
- Performance test expects `calculate_returns_and_momentum()` to run under ~5 seconds for 10K rows; keep that benchmark in mind when modifying indicator calculations.
- Edge-case tests confirm the pipeline raises on missing columns or empty data, so new features should preserve these failure modes rather than swallow them.

## Resources & documentation to reference
- For narrative context, consult `README.md`, `src/trading_strategy/README.md`, and `MODULARIZATION_SUMMARY.md` to understand why the notebook split into modules and how they expect to evolve.
- When crafting new strategies, inspect `strategies/README.md` and the YAML files themselves to match structure, naming, and combination methods before touching code.
- Use `examples/basic_usage.py` as the regression guard: any change to modules should still allow that script to run without editing its core sequence of steps.
