"""
Módulo de backtesting
Ejecuta backtests de estrategias y calcula métricas de performance.

El motor trabaja sobre arrays numpy (sin copiar el DataFrame completo por
experimento) y extrae los trades de forma vectorizada.
"""

import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Dict, List, Optional, Any
from .indicators import compute_signal, compute_combined_signal, new_value_cache
from .constants import (
    COL_SIGNAL, COL_OPEN, COL_CLOSE, COL_HIGH, COL_LOW, COL_VOLUME
)
from .utils.stats import calculate_trade_statistics
from .validation_metrics import compute_validation_metrics, ValidationContext


# ============================================================================
# MOTOR VECTORIZADO
# ============================================================================

def _target_position(signal: np.ndarray, position_type: str) -> np.ndarray:
    """Convierte el array de señal en la posición objetivo según el tipo."""
    if position_type == 'long':
        return np.where(signal > 0, 1.0, 0.0)
    elif position_type == 'short':
        return np.where(signal < 0, -1.0, 0.0)
    else:  # both
        return np.asarray(signal, dtype=float)


def _apply_costs(entry_price, exit_price, is_long, commission, slippage):
    """
    Calcula PnL bruto y neto de costes (comisión + slippage) por trade.

    Modelo (round-trip):
    - ``commission`` = fracción del notional, cobrada en entrada Y salida (×2).
    - ``slippage`` = fracción del precio, adversa en ambos lados (compras más caro,
      vendes más barato).

    Returns
    -------
    dict con arrays: gross_pnl_pct, slippage_pct, commission_pct, costs_pct, net_pnl_pct
    """
    gross = np.where(is_long,
                     (exit_price - entry_price) / entry_price,
                     (entry_price - exit_price) / entry_price)

    if commission == 0.0 and slippage == 0.0:
        zeros = np.zeros_like(gross)
        return {'gross_pnl_pct': gross, 'slippage_pct': zeros,
                'commission_pct': zeros, 'costs_pct': zeros, 'net_pnl_pct': gross}

    # Precios efectivos tras slippage (adverso en ambos lados)
    eff_entry = np.where(is_long, entry_price * (1 + slippage), entry_price * (1 - slippage))
    eff_exit = np.where(is_long, exit_price * (1 - slippage), exit_price * (1 + slippage))
    gross_slip = np.where(is_long,
                          (eff_exit - eff_entry) / eff_entry,
                          (eff_entry - eff_exit) / eff_entry)

    slippage_pct = gross - gross_slip               # retorno perdido por slippage
    commission_pct = np.full_like(gross, 2.0 * commission)
    costs_pct = slippage_pct + commission_pct
    net = gross - costs_pct
    return {'gross_pnl_pct': gross, 'slippage_pct': slippage_pct,
            'commission_pct': commission_pct, 'costs_pct': costs_pct, 'net_pnl_pct': net}


def _extract_trades_raw(
    signal: np.ndarray,
    open_np: np.ndarray,
    close_np: np.ndarray,
    position_type: str = 'long',
    use_next_open: bool = True,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Extrae los trades cerrados de forma vectorizada (con costes opcionales).

    Replica la semántica del antiguo bucle:
    - Si ``use_next_open``, la señal en T se ejecuta al Open de T+1.
    - Un trade es un tramo maximal de posición no nula; entra al precio de la vela
      donde empieza el tramo y sale al precio de la vela del siguiente cambio.
    - La posición abierta al final (sin cambio posterior) se ignora.

    ``commission``/``slippage`` (fracciones) se aplican al PnL; con ambos a 0 el
    ``pnl_pct`` es bruto (comportamiento previo).

    Returns
    -------
    dict con arrays (longitud n_trades, posiblemente 0): entry_idx, exit_idx,
    entry_price, exit_price, is_long, pnl_pct (neto), gross_pnl_pct, slippage_pct,
    commission_pct, costs_pct.
    """
    target = _target_position(signal, position_type)

    if use_next_open:
        prices = open_np
        exec_pos = np.empty_like(target)
        exec_pos[0] = 0.0
        exec_pos[1:] = target[:-1]
    else:
        prices = close_np
        exec_pos = target

    # Puntos de cambio (diff[0] = 0, igual que df.diff().fillna(0))
    diff = np.empty_like(exec_pos)
    diff[0] = 0.0
    diff[1:] = exec_pos[1:] - exec_pos[:-1]
    change_idx = np.nonzero(diff != 0)[0]

    empty = np.array([], dtype=float)
    empty_i = np.array([], dtype=int)
    if change_idx.size < 2:
        return {'entry_idx': empty_i, 'exit_idx': empty_i,
                'entry_price': empty, 'exit_price': empty,
                'is_long': np.array([], dtype=bool), 'pnl_pct': empty,
                'gross_pnl_pct': empty, 'slippage_pct': empty,
                'commission_pct': empty, 'costs_pct': empty}

    vals = exec_pos[change_idx]
    prices_at = prices[change_idx]

    # Emparejar cada cambio con el siguiente (apertura -> cierre)
    entry_vals = vals[:-1]
    entry_price = prices_at[:-1]
    exit_price = prices_at[1:]
    entry_idx = change_idx[:-1]
    exit_idx = change_idx[1:]

    mask = entry_vals != 0
    entry_vals = entry_vals[mask]
    entry_price = entry_price[mask]
    exit_price = exit_price[mask]
    entry_idx = entry_idx[mask]
    exit_idx = exit_idx[mask]

    is_long = entry_vals > 0
    costs = _apply_costs(entry_price, exit_price, is_long, commission, slippage)

    return {'entry_idx': entry_idx, 'exit_idx': exit_idx,
            'entry_price': entry_price, 'exit_price': exit_price,
            'is_long': is_long, 'pnl_pct': costs['net_pnl_pct'],
            'gross_pnl_pct': costs['gross_pnl_pct'], 'slippage_pct': costs['slippage_pct'],
            'commission_pct': costs['commission_pct'], 'costs_pct': costs['costs_pct']}


def extract_trade_returns(
    signal: np.ndarray,
    open_np: np.ndarray,
    close_np: np.ndarray,
    position_type: str = 'long',
    use_next_open: bool = True,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> np.ndarray:
    """Devuelve solo el array de retornos netos (pnl_pct) de los trades cerrados."""
    return _extract_trades_raw(signal, open_np, close_np, position_type,
                               use_next_open, commission, slippage)['pnl_pct']


def _signal_array(
    df: pd.DataFrame,
    indicator: Optional[str],
    params: Dict[str, Any],
    indicators_combo: Optional[List[Dict[str, Any]]],
    combination_method: str,
    value_cache: Optional["OrderedDict"] = None,
) -> Optional[np.ndarray]:
    """
    Calcula el array de señal para un indicador individual o un combo.

    ``value_cache`` memoiza las llamadas costosas a pandas-ta (reutilizadas entre
    distintos thresholds y position_types del mismo grid search).
    """
    volume_s = df[COL_VOLUME] if COL_VOLUME in df.columns else None
    if indicators_combo is not None:
        return compute_combined_signal(
            df[COL_OPEN], df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], volume_s,
            indicators_combo, combination_method, value_cache=value_cache
        )
    if indicator is None:
        return None
    sig, _ = compute_signal(
        df[COL_OPEN], df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], volume_s,
        indicator, params, want_aux=False, value_cache=value_cache
    )
    return sig


# ============================================================================
# API PÚBLICA
# ============================================================================

def backtest_strategy(
    df: pd.DataFrame,
    indicator: Optional[str],
    params: Dict[str, Any],
    position_type: str = 'long',
    indicators_combo: Optional[List[Dict[str, Any]]] = None,
    combination_method: str = 'AND',
    initial_capital: float = 10000.0,
    commission: float = 0.001,   # fracción del notional, por lado (×2 round-trip)
    slippage: float = 0.0001,    # fracción del precio, adversa en ambos lados
    use_next_open: bool = True,
    value_cache: Optional["OrderedDict"] = None,
    validation_context: Optional[ValidationContext] = None,
) -> Optional[Dict[str, float]]:
    """
    Backtest de estrategia de trading con cálculo de métricas.

    Ruta rápida: calcula la señal, extrae los retornos NETOS de los trades en numpy
    y calcula las métricas, sin construir el DataFrame de trades.

    ``commission``/``slippage`` (fracciones) se aplican al PnL; las métricas quedan
    netas de costes y se añaden agregados nominales (total_costs_usd, etc.).

    ``value_cache`` (opcional) reutiliza el resultado del indicador (pandas-ta)
    entre los distintos thresholds y ``position_type`` del mismo grid search.

    ``validation_context`` (opcional) añade las métricas de validación
    anti-overfitting (sharpe_train/val, sub-ventanas, régimen, IR vs B&H)
    calculadas con la MISMA señal — ver validation_metrics.build_validation_context.
    Son métricas por-período (close-to-close), complementarias de las por-trade.

    Returns
    -------
    dict : Métricas de performance o None si no hay trades.
    """
    signal = _signal_array(df, indicator, params, indicators_combo, combination_method, value_cache)
    if signal is None:
        return None

    open_np = df[COL_OPEN].to_numpy()
    close_np = df[COL_CLOSE].to_numpy()
    raw = _extract_trades_raw(signal, open_np, close_np, position_type,
                              use_next_open, commission, slippage)

    if raw['pnl_pct'].size == 0:
        return None

    metrics = calculate_trade_statistics(
        raw['pnl_pct'], initial_capital=initial_capital,
        costs_pct=raw['costs_pct'], commission_pct=raw['commission_pct'],
        slippage_pct=raw['slippage_pct'],
    )

    metrics.update({
        'strategy_name': f"{indicator}_{params}" if indicator else "combo_strategy",
        'params': params if params else {},
        'n_transactions': metrics['total_trades'] * 2  # Estimado
    })

    if validation_context is not None:
        pos = _target_position(signal, position_type)
        metrics.update(compute_validation_metrics(pos, validation_context))

    return metrics


def get_strategy_trades(
    df: pd.DataFrame,
    indicator: Optional[str],
    params: Dict[str, Any],
    position_type: str = 'long',
    indicators_combo: Optional[List[Dict[str, Any]]] = None,
    combination_method: str = 'AND',
    use_next_open: bool = True,
    commission: float = 0.001,
    slippage: float = 0.0001,
) -> pd.DataFrame:
    """
    Ejecuta la estrategia y extrae la lista detallada de operaciones (trades).

    ``commission``/``slippage`` (fracciones) se aplican: ``pnl_pct`` es neto y se
    incluyen columnas de costes efectivos para análisis.

    Returns
    -------
    DataFrame : entry_time, exit_time, type, entry_price, exit_price, pnl_pct (neto),
    gross_pnl_pct, slippage_pct, commission_pct, costs_pct, duration.
    """
    signal = _signal_array(df, indicator, params, indicators_combo, combination_method)
    if signal is None:
        return pd.DataFrame()

    open_np = df[COL_OPEN].to_numpy()
    close_np = df[COL_CLOSE].to_numpy()
    raw = _extract_trades_raw(signal, open_np, close_np, position_type,
                              use_next_open, commission, slippage)

    if raw['pnl_pct'].size == 0:
        return pd.DataFrame()

    index = df.index
    entry_time = index[raw['entry_idx']]
    exit_time = index[raw['exit_idx']]

    trades = pd.DataFrame({
        'entry_time': entry_time,
        'exit_time': exit_time,
        'type': np.where(raw['is_long'], 'long', 'short'),
        'entry_price': raw['entry_price'],
        'exit_price': raw['exit_price'],
        'pnl_pct': raw['pnl_pct'],
        'gross_pnl_pct': raw['gross_pnl_pct'],
        'slippage_pct': raw['slippage_pct'],
        'commission_pct': raw['commission_pct'],
        'costs_pct': raw['costs_pct'],
        'duration': exit_time - entry_time,
    })
    return trades
