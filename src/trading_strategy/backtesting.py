"""
Módulo de backtesting
Ejecuta backtests de estrategias y calcula métricas de performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from .indicators import calculate_indicator_and_signals, combine_indicator_signals
from .constants import COL_SIGNAL, COL_RETURNS, COL_FUTURE_RET, COL_OPEN, COL_CLOSE


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
    # Calcular señales según tipo de estrategia
    if indicators_combo is not None:
        # Estrategia multi-indicador
        df = combine_indicator_signals(df, indicators_combo, combination_method, inplace=True)
    else:
        # Estrategia de indicador individual
        if indicator is None:
             return None
        df = calculate_indicator_and_signals(df, indicator, params, inplace=True)
    
    if COL_SIGNAL not in df.columns or df[COL_SIGNAL].isna().all():
        # print("Debug: No signal column or all NaN")
        return None
    
    # --- Lógica de Ejecución y Costes ---
    
    # 1. Determinar Posición Objetivo
    # Filtrar señales según position_type
    raw_signal = df[COL_SIGNAL].fillna(0)
    
    if position_type == 'long':
        # Solo tomar señales positivas
        target_position = np.where(raw_signal > 0, 1, 0)
    elif position_type == 'short':
        # Solo tomar señales negativas
        target_position = np.where(raw_signal < 0, -1, 0)
    else: # both
        # Mantener señales tal cual (-1, 0, 1)
        target_position = raw_signal.values
        
    target_position = pd.Series(target_position, index=df.index)
    
    # 2. Calcular Retornos de la Estrategia
    if use_next_open:
        # Ejecución al Open de la siguiente vela (T+1)
        # La posición en T se basa en la señal de T-1
        current_position = target_position.shift(1).fillna(0)
        
        # Retorno del periodo T: (Open[T+1] - Open[T]) / Open[T]
        # Representa mantener la posición desde Open[T] hasta Open[T+1]
        next_open = df[COL_OPEN].shift(-1)
        period_returns = (next_open - df[COL_OPEN]) / df[COL_OPEN]
        
    else:
        # Ejecución al Close de la vela actual (T) - Optimista
        # La posición en T se asume instantánea al cierre
        current_position = target_position
        
        # Retorno del periodo T: (Close[T+1] - Close[T]) / Close[T]
        # Usamos future_ret+1 que ya está calculado o lo calculamos
        target_col = f'{COL_FUTURE_RET}+1'
        if target_col in df.columns:
            period_returns = df[target_col]
        else:
            period_returns = df[COL_CLOSE].shift(-1) / df[COL_CLOSE] - 1

    # Retorno Bruto = Posición * Retorno del Mercado
    gross_returns = current_position * period_returns
    
    # 3. Calcular Costes de Transacción
    # Turnover: Cambio absoluto en la posición
    turnover = current_position.diff().abs().fillna(0)
    
    # Corregir el primer trade (si empieza con posición != 0)
    if current_position.iloc[0] != 0:
        turnover.iloc[0] = abs(current_position.iloc[0])
        
    costs = turnover * (commission + slippage)
    
    # 4. Retorno Neto
    net_returns = gross_returns - costs
    
    # Eliminar NaNs (generados por shifts)
    valid_idx = ~np.isnan(net_returns) & ~np.isnan(period_returns)
    returns = net_returns[valid_idx]
    
    if len(returns) < 10:
        return None

    # Calcular métricas
    # returns ya es una Series con los retornos netos
    cumulative_returns = (1 + returns).cumprod()
    
    # Liberar arrays temporales
    # del signal_array, future_ret_array, mask, adjusted_returns # Ya no existen
    
    total_return = cumulative_returns.iloc[-1] - 1
    n_periods = len(returns)
    
    # Calcular número real de trades (basado en turnover)
    # Un trade completo (entrada + salida) implica turnover = 2 (aprox)
    # O simplemente contar transacciones (turnover > 0)
    n_transactions = (turnover[valid_idx] > 0).sum()
    # Estimación de trades cerrados (transactions / 2)
    n_trades = max(1, int(n_transactions / 2))
    
    # Hit rate (Periodos positivos vs totales)
    hit_rate = (returns > 0).mean()
    
    # Sharpe Ratio (anualizado aproximado)
    # Asumiendo datos horarios (24*365 = 8760) o diarios (365)
    # Para ser agnóstico, usamos sharpe por periodo y el usuario escala si quiere
    # Ojo: Si returns.std() es 0, sharpe es 0
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Max Drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win/Loss stats (basado en periodos, no trades completos en vectorizado simple)
    # Para estadísticas por Trade real, se necesitaría un loop de eventos.
    # Aquí mantenemos estadísticas por periodo (barra) que es estándar en vectorizado
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0 # Win rate de periodos
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Profit Factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Risk-Reward Ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Calmar Ratio (retorno / max drawdown)
    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Sortino Ratio (penaliza solo volatilidad bajista)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
    sortino = returns.mean() / downside_std if downside_std > 0 else 0
    
    # --- Métricas Avanzadas (SQN, Kelly) ---
    # SQN (System Quality Number) - Aproximación basada en periodos
    # SQN = sqrt(N) * (Mean / StdDev)
    # Nota: Para SQN real se necesitan retornos por trade, aquí usamos retornos por periodo
    # lo cual escala con sqrt(N_periods). Es una métrica de calidad de la curva de equity.
    sqn = np.sqrt(len(returns)) * sharpe
    
    # Kelly Criterion (f = p - q/b)
    # p = win_rate, q = 1-p, b = risk_reward_ratio
    # Usamos win_rate de periodos y risk_reward de periodos
    if risk_reward > 0:
        kelly = win_rate - (1 - win_rate) / risk_reward
    else:
        kelly = 0
        
    # Expectancy (Esperanza matemática por periodo)
    # E = (Win% * AvgWin) - (Loss% * AvgLoss)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
    
    # --- Métricas Nominales (USD) ---
    # Reindexar equity curve al índice original para alinear con turnover si fuera necesario
    # Pero cumulative_returns ya tiene el índice filtrado
    equity_curve = initial_capital * cumulative_returns
    final_equity = equity_curve.iloc[-1]
    net_profit_usd = final_equity - initial_capital
    
    # Max Drawdown Nominal (Peak - Valley en $)
    running_max_equity = equity_curve.expanding().max()
    drawdown_usd = running_max_equity - equity_curve
    max_drawdown_usd = drawdown_usd.max()
    
    # Costes totales en USD (aproximado)
    # costs es % del capital en ese momento.
    # Aproximación: costs_usd = sum(costs * equity_curve.shift(1))
    # Esto es complejo vectorizado exacto.
    # Usamos una métrica simple:
    total_costs_pct = costs[valid_idx].sum()
    
    metrics = {
        'total_return': float(total_return),
        'n_trades': int(n_trades), # Ahora es estimado de trades reales
        'n_transactions': int(n_transactions),
        'hit_rate': float(hit_rate),
        'win_rate': float(win_rate),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'calmar_ratio': float(calmar),
        'max_drawdown': float(max_drawdown),
        'profit_factor': float(profit_factor),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'risk_reward_ratio': float(risk_reward),
        'best_trade': float(returns.max()),
        'worst_trade': float(returns.min()),
        'avg_return_per_trade': float(returns.mean()), # Retorno promedio por periodo
        'volatility': float(returns.std()),
        'sqn': float(sqn),
        'kelly_criterion': float(kelly),
        'expectancy': float(expectancy),
        
        # Métricas Nominales
        'net_profit_usd': float(net_profit_usd),
        'final_equity': float(final_equity),
        'max_drawdown_usd': float(max_drawdown_usd),
        'total_costs_pct': float(total_costs_pct)
    }
    
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
