"""
Tests del módulo de validación anti-overfitting (validation_metrics).
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from trading_strategy.validation_metrics import (
    asymmetric_stability,
    backtest_with_validation,
    build_validation_context,
    compute_stability_table,
    compute_validation_metrics,
    decision_rule,
    dedupe_runs,
    periods_per_year_from_timeframe,
    sharpe,
    split_train_val,
)
from trading_strategy.grid_search import strategy_grid_search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_df():
    """1000 barras diarias con drift positivo (random walk geométrico)."""
    rng = np.random.default_rng(7)
    n = 1000
    ret = rng.normal(0.0008, 0.02, n)
    close = 100.0 * np.cumprod(1.0 + ret)
    dates = pd.date_range('2022-01-01', periods=n, freq='1D')
    return pd.DataFrame({
        'Open': close * (1 + rng.normal(0, 0.001, n)),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': rng.integers(100, 1000, n).astype(float),
    }, index=dates)


# ---------------------------------------------------------------------------
# [S1] Stability asimétrica — los 3 casos canónicos
# ---------------------------------------------------------------------------

def test_stability_degradation_penalized():
    # MACD: 1º en train → 10º en validación (de 15) — overfitting canónico
    s = asymmetric_stability(rank_train=1, rank_val=10, n_strategies=15,
                             sharpe_val=1.0)
    assert s == pytest.approx(1.0 - 9 / 14)  # ≈ 0.357


def test_stability_improvement_not_penalized():
    # Fees momentum: 9º en train → 1º en validación — evidencia A FAVOR
    s = asymmetric_stability(rank_train=9, rank_val=1, n_strategies=15,
                             sharpe_val=1.0)
    assert s == 1.0


def test_stability_mediocrity_gated():
    # 14º → 14º con Sharpe bajo: la consistencia en la mediocridad no es señal
    s = asymmetric_stability(rank_train=14, rank_val=14, n_strategies=15,
                             sharpe_val=0.2)
    assert s == 0.0


# ---------------------------------------------------------------------------
# Timeframes y split
# ---------------------------------------------------------------------------

def test_periods_per_year_from_timeframe():
    assert periods_per_year_from_timeframe('1d') == 365
    assert periods_per_year_from_timeframe('4h') == 6 * 365
    assert periods_per_year_from_timeframe('1H') == 24 * 365
    with pytest.raises(KeyError):
        periods_per_year_from_timeframe('3h')


def test_split_train_val_chronological():
    idx = pd.date_range('2023-01-01', periods=100, freq='1D')
    mtr, mval = split_train_val(idx, val_frac=0.3)
    assert mtr.sum() == 70 and mval.sum() == 30
    assert mtr[:70].all() and mval[70:].all()  # corte cronológico, no aleatorio


# ---------------------------------------------------------------------------
# Wrapper y hot path
# ---------------------------------------------------------------------------

def test_always_long_matches_buy_and_hold(daily_df):
    """Posición constante = B&H sin costes (no hay turnover tras la entrada)."""
    pos = pd.Series(1.0, index=daily_df.index)
    m = backtest_with_validation(daily_df, pos, commission_rate=0.0,
                                 slippage_rate=0.0, signal_delay=1)
    ret = daily_df['Close'].pct_change().fillna(0.0)
    _, mask_val = split_train_val(daily_df.index, 0.3)
    bh_val = ret[mask_val]
    # signal_delay=1 solo desplaza la primera barra (posición constante)
    assert m['sharpe_val'] == pytest.approx(sharpe(bh_val), rel=1e-6)
    assert m['roi_val'] == pytest.approx(m['roi_bh_val'], rel=1e-9)
    assert m['n_trades_total'] == 0  # sin cambios de posición en validación


def test_no_lookahead_delay_applied(daily_df):
    """Señal 'oráculo' (larga cuando el retorno de ESA barra es positivo):
    con delay=0 captura el retorno de la misma barra (lookahead puro);
    con delay>=1 el shift destruye la ventaja — verifica que el delay se aplica."""
    ret = daily_df['Close'].pct_change().fillna(0.0)
    oracle = pd.Series((ret > 0).astype(float), index=daily_df.index)

    m0 = backtest_with_validation(daily_df, oracle, commission_rate=0.0,
                                  slippage_rate=0.0, signal_delay=0)
    m1 = backtest_with_validation(daily_df, oracle, commission_rate=0.0,
                                  slippage_rate=0.0, signal_delay=1)
    assert m0['sharpe_val'] > 5.0            # el oráculo sin delay es absurdo
    assert m1['sharpe_val'] < m0['sharpe_val'] * 0.5


def test_costs_reduce_returns(daily_df):
    """Con turnover, añadir costes debe reducir el ROI de validación."""
    rng = np.random.default_rng(3)
    pos = pd.Series(rng.integers(0, 2, len(daily_df)).astype(float),
                    index=daily_df.index)
    m_free = backtest_with_validation(daily_df, pos, commission_rate=0.0,
                                      slippage_rate=0.0)
    m_cost = backtest_with_validation(daily_df, pos, commission_rate=0.001,
                                      slippage_rate=0.001)
    assert m_cost['roi_val'] < m_free['roi_val']
    assert m_cost['n_trades_total'] > 15  # señal aleatoria: mucho turnover


def test_context_and_wrapper_equivalent(daily_df):
    """compute_validation_metrics(pos, ctx) == backtest_with_validation(...)"""
    rng = np.random.default_rng(11)
    pos = pd.Series(rng.integers(0, 2, len(daily_df)).astype(float),
                    index=daily_df.index)
    ctx = build_validation_context(daily_df, commission=0.001, slippage=0.001,
                                   periods_per_year=365)
    m_ctx = compute_validation_metrics(pos.to_numpy(), ctx)
    m_wrap = backtest_with_validation(daily_df, pos, commission_rate=0.001,
                                      slippage_rate=0.001)
    for k, v in m_wrap.items():
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(m_ctx[k]), k
        else:
            assert m_ctx[k] == pytest.approx(v, rel=1e-12), k


def test_short_dataset_subwindows_nan():
    """n_val < 90 → sub-ventanas NaN (mínimo 30 obs/ventana), sin excepción."""
    rng = np.random.default_rng(5)
    n = 200  # n_val = 60
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    df = pd.DataFrame({'Close': close},
                      index=pd.date_range('2023-01-01', periods=n, freq='1D'))
    m = backtest_with_validation(df, pd.Series(1.0, index=df.index))
    assert np.isnan(m['sharpe_val_w1'])
    assert not np.isnan(m['sharpe_val'])


# ---------------------------------------------------------------------------
# Dedupe y regla de decisión
# ---------------------------------------------------------------------------

def test_dedupe_runs_keeps_latest():
    df = pd.DataFrame({
        'experiment_id': ['abc', 'abc', 'def'],
        'ticker': ['BTCUSDT'] * 3,
        'timeframe': ['1d'] * 3,
        'session_id': ['20260101_000000', '20260201_000000', '20260101_000000'],
        'sharpe_val': [0.1, 0.9, 0.5],
    })
    out = dedupe_runs(df)
    assert len(out) == 2
    assert out.attrs['duplicates_removed'] == 1
    kept = out[out['experiment_id'] == 'abc'].iloc[0]
    assert kept['session_id'] == '20260201_000000'  # el más reciente


def _base_row(**over):
    row = {'sharpe_val': 1.0, 'stability': 0.9, 'min_regime_sharpe': 0.2,
           'passes_ir_2of3': 1.0, 'passes_min_trades': 1.0}
    row.update(over)
    return pd.Series(row)


def test_decision_rule_three_branches():
    assert decision_rule(_base_row(), 0.8, True) == 'SELECCIONAR'
    # Falla 2 criterios (régimen + comisión) con stability >= 0.6 → ZONA_GRIS
    assert decision_rule(_base_row(min_regime_sharpe=-0.1), 0.8, False) == 'ZONA_GRIS'
    # Sharpe insuficiente → DESCARTAR
    assert decision_rule(_base_row(sharpe_val=0.2), 0.8, True) == 'DESCARTAR'


def test_stability_table_columns():
    df = pd.DataFrame({
        'ticker': ['BTC', 'BTC', 'ETH', 'ETH'],
        'strategy_name': ['a', 'b', 'a', 'b'],
        'sharpe_train': [1.0, 0.5, 0.8, 0.9],
        'sharpe_val': [0.9, 0.6, 0.7, 0.2],
    })
    out = compute_stability_table(df)
    assert {'rank_train', 'rank_val', 'n_universe', 'stability'} <= set(out.columns)
    assert (out['n_universe'] == 2).all()  # universo por activo


# ---------------------------------------------------------------------------
# Integración con el grid search (Parquet + manifest)
# ---------------------------------------------------------------------------

VAL_COLS = {'sharpe_train', 'sharpe_val', 'sr_val_period', 'sharpe_val_w1',
            'roi_val', 'roi_bh_val', 'maxdd_val', 'min_regime_sharpe',
            'passes_ir_2of3', 'n_trades_total'}


def test_grid_search_emits_validation_columns(daily_df, tmp_path):
    cfg = [{'name': 'rsi_test', 'indicator': 'rsi',
            'params_grid': {'period': [7, 14], 'overbought': [70],
                            'oversold': [30]}}]
    res = strategy_grid_search(daily_df, cfg, ticker='BTCUSDT', timeframe='1d',
                               experiment_name='valtest', verbose=False,
                               output_dir=tmp_path)
    assert not res.empty
    assert VAL_COLS <= set(res.columns)

    parquets = list(tmp_path.glob('results_*.parquet'))
    assert len(parquets) == 1
    stored = pd.read_parquet(parquets[0])
    assert VAL_COLS <= set(stored.columns)

    manifest = json.loads(next(tmp_path.glob('manifest_*.json')).read_text(encoding='utf-8'))
    assert manifest['val_frac'] == 0.3
    assert manifest['signal_delay'] == 1
    assert manifest['periods_per_year'] == 365
    assert manifest['n_experiments_enumerated'] == 2


def test_grid_search_validation_off(daily_df):
    cfg = [{'name': 'rsi_test', 'indicator': 'rsi',
            'params_grid': {'period': [14], 'overbought': [70],
                            'oversold': [30]}}]
    res = strategy_grid_search(daily_df, cfg, ticker='BTCUSDT', timeframe='1d',
                               verbose=False, validation=False)
    assert not res.empty
    assert 'sharpe_val' not in res.columns
