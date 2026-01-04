"""
Módulo de backtesting
Ejecuta backtests de estrategias y calcula métricas de performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from .indicators import calculate_indicator_and_signals, combine_indicator_signals
from .constants import COL_SIGNAL, COL_RETURNS, COL_FUTURE_RET, COL_OPEN, COL_CLOSE
from .utils.stats import calculate_trade_statistics


def backtest_strategy(
    df: pd.DataFrame, 
    indicator: Optional[str], 
    params: Dict[str, Any], 
    position_type: str = 'long', 
    indicators_combo: Optional[List[Dict[str, Any]]] = None, 
    combination_method: str = 'AND', 
    initial_capital: float = 10000.0,
    commission: float = 0.001,  # 0.1% per trade
    slippage: float = 0.0001,   # 0.01% slippage
    use_next_open: bool = True  # Execute at next Open
) -> Optional[Dict[str, float]]:
    """
    Backtest de estrategia de trading con cálculo de métricas
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con datos OHLCV y columna 'returns'
    indicator : str
        Nombre del indicador ('rsi', 'macd', 'willr', etc.)
        Si indicators_combo es provisto, este parámetro se ignora
    params : dict
        Parámetros de la estrategia (period, thresholds, etc.)
        Solo usado si indicator es un string (indicador individual)
    position_type : str
        'long', 'short' o 'both'
    indicators_combo : list or None
        Lista de configuraciones para estrategia multi-indicador
    combination_method : str
        Método de combinación si indicators_combo es usado
    initial_capital : float
        Capital inicial para calcular métricas nominales (USD)
        Default: 10000.0
    commission : float
        Comisión por operación (ej: 0.001 = 0.1%)
    slippage : float
        Deslizamiento estimado (ej: 0.0001 = 0.01%)
    use_next_open : bool
        Si True, ejecuta operaciones al Open de la siguiente vela (más realista).
        Si False, asume ejecución al Close de la vela de señal (optimista).
    
    Returns:
    --------
    dict : Métricas de performance o None si falla
    """
    # 1. Obtener Trades
    trades = get_strategy_trades(
        df, 
        indicator, 
        params, 
        position_type, 
        indicators_combo, 
        combination_method, 
        use_next_open
    )
    
    if trades.empty:
        return None
        
    # 2. Calcular Métricas usando el módulo centralizado
    metrics = calculate_trade_statistics(
        trades, 
        returns_col='pnl_pct', 
        initial_capital=initial_capital
    )
    
    # 3. Agregar metadatos
    metrics.update({
        'strategy_name': f"{indicator}_{params}" if indicator else "combo_strategy",
        'params': params if params else {},
        'n_transactions': metrics['total_trades'] * 2  # Estimado
    })
    
    return metrics


def get_strategy_trades(
    df: pd.DataFrame, 
    indicator: Optional[str], 
    params: Dict[str, Any], 
    position_type: str = 'long', 
    indicators_combo: Optional[List[Dict[str, Any]]] = None, 
    combination_method: str = 'AND',
    use_next_open: bool = True
) -> pd.DataFrame:
    """
    Ejecuta la estrategia y extrae la lista detallada de operaciones (trades)
    
    Parameters:
    -----------
    Mismos parámetros que backtest_strategy
    
    Returns:
    --------
    DataFrame : Lista de trades con entry_time, exit_time, prices, pnl, etc.
    """
    # Trabajar sobre copia para no afectar el DF original
    df = df.copy()
    
    # 1. Calcular Señales
    if indicators_combo is not None:
        df = combine_indicator_signals(df, indicators_combo, combination_method, inplace=True)
    else:
        if indicator is None:
             return pd.DataFrame()
        df = calculate_indicator_and_signals(df, indicator, params, inplace=True)
    
    if COL_SIGNAL not in df.columns:
        return pd.DataFrame()

    # 2. Determinar Posición Objetivo
    raw_signal = df[COL_SIGNAL].fillna(0)
    
    if position_type == 'long':
        target_position = np.where(raw_signal > 0, 1, 0)
    elif position_type == 'short':
        target_position = np.where(raw_signal < 0, -1, 0)
    else: # both
        target_position = raw_signal.values
        
    # 3. Extraer Trades
    trades = []
    current_pos = 0
    entry_price = 0.0
    entry_time = None
    
    # Definir precios de ejecución
    if use_next_open:
        # Si usamos next open, la señal en T se ejecuta al Open de T+1
        # Alineamos la posición de ejecución: exec_pos[T] es la posición que tenemos durante la vela T
        # que fue determinada por la señal en T-1.
        # Pero para detectar el cambio, miramos cuando cambia la posición deseada.
        
        # Precios de ejecución son los Open
        prices = df[COL_OPEN]
        
        # La posición efectiva cambia al Open de T+1 basado en señal de T
        # Shift 1 para alinear: exec_position[t] es la posición tomada al inicio de t
        exec_position = pd.Series(target_position, index=df.index).shift(1).fillna(0)
    else:
        # Ejecución al Close
        prices = df[COL_CLOSE]
        exec_position = pd.Series(target_position, index=df.index)
        
    # Identificar cambios de posición
    # diff[t] != 0 significa que la posición cambió en t (al precio de t)
    diffs = exec_position.diff().fillna(0)
    changes = diffs[diffs != 0]
    
    # Iterar solo sobre los cambios para reconstruir trades
    for time_idx, change in changes.items():
        new_pos = exec_position[time_idx]  # pyright: ignore[reportCallIssue, reportArgumentType]
        price = prices[time_idx] # pyright: ignore[reportArgumentType, reportCallIssue]
        
        # 1. Cierre de posición existente
        if current_pos != 0:
            # Si pasamos a flat (0) o invertimos posición (signo opuesto)
            if new_pos == 0 or np.sign(new_pos) != np.sign(current_pos):
                
                # Calcular PnL
                if current_pos > 0: # Long exit
                    pnl_pct = (price - entry_price) / entry_price
                    trade_type = 'long'
                else: # Short exit
                    pnl_pct = (entry_price - price) / entry_price
                    trade_type = 'short'
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time_idx,
                    'type': trade_type,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl_pct': pnl_pct,
                    'duration': time_idx - entry_time if entry_time else None # pyright: ignore[reportOperatorIssue]
                })
                
                current_pos = 0
        
        # 2. Apertura de nueva posición
        if new_pos != 0:
            # Si estábamos flat o acabamos de cerrar (inversión)
            if current_pos == 0:
                current_pos = new_pos
                entry_price = price
                entry_time = time_idx
    
    # Si queda posición abierta al final, se puede cerrar o ignorar
    # Aquí la ignoramos para coincidir con métricas de trades cerrados
    
    return pd.DataFrame(trades)
