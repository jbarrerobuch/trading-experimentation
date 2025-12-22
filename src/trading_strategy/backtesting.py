"""
Módulo de backtesting
Ejecuta backtests de estrategias y calcula métricas de performance
"""

import numpy as np
import pandas as pd
from .indicators import calculate_indicator_and_signals, combine_indicator_signals


def backtest_strategy(df, indicator, params, position_type='long', indicators_combo=None, 
                      combination_method='AND', initial_capital=10000.0):
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
        Ejemplo:
        [
            {'indicator': 'rsi', 'params': {'period': 14, 'overbought': 70, 'oversold': 30}},
            {'indicator': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9}}
        ]
    combination_method : str
        Método de combinación si indicators_combo es usado
        Opciones: 'AND', 'OR', 'MAJORITY', 'WEIGHTED', 'UNANIMOUS_LONG', 'UNANIMOUS_SHORT'
    initial_capital : float
        Capital inicial para calcular métricas nominales (USD)
        Default: 10000.0
    
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
        df = calculate_indicator_and_signals(df, indicator, params, inplace=True)
    
    if 'signal' not in df.columns or df['signal'].isna().all():
        return None
    
    # future_ret ya está precalculado por calculate_returns_and_momentum()
    # Verificar que existe la columna (ahora se llama future_ret+1)
    target_col = 'future_ret+1'
    if target_col not in df.columns:
        if 'future_ret' in df.columns:
             target_col = 'future_ret'
        else:
             df[target_col] = df['returns'].shift(-1)
    
    # Optimización: Usar arrays numpy para operaciones vectorizadas (más rápido)
    signal_array = df['signal'].values
    future_ret_array = df[target_col].values
    
    # Filtrar por tipo de posición y ajustar retornos
    if position_type == 'long':
        mask = signal_array > 0
        adjusted_returns = future_ret_array[mask]
    elif position_type == 'short':
        mask = signal_array < 0
        adjusted_returns = -future_ret_array[mask]  # Invertir retornos para short
    else:  # both
        mask = signal_array != 0
        # Invertir retornos donde signal < 0
        adjusted_returns = np.where(signal_array[mask] < 0, 
                                     -future_ret_array[mask], 
                                     future_ret_array[mask])
    
    # Eliminar NaN y crear Series
    adjusted_returns = adjusted_returns[~np.isnan(adjusted_returns)]
    
    if len(adjusted_returns) < 10:
        # Liberar memoria antes de retornar
        del signal_array, future_ret_array, mask, adjusted_returns
        return None
    
    # Calcular métricas
    returns = pd.Series(adjusted_returns)
    cumulative_returns = (1 + returns).cumprod()
    
    # Liberar arrays temporales
    del signal_array, future_ret_array, mask, adjusted_returns
    
    total_return = cumulative_returns.iloc[-1] - 1
    n_trades = len(returns)
    
    # Hit rate
    hit_rate = (returns > 0).mean()
    
    # Sharpe Ratio (anualizado aproximado)
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Max Drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win/Loss stats
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
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
    
    # --- Métricas Nominales (USD) ---
    equity_curve = initial_capital * cumulative_returns
    final_equity = equity_curve.iloc[-1]
    net_profit_usd = final_equity - initial_capital
    
    # Max Drawdown Nominal (Peak - Valley en $)
    running_max_equity = equity_curve.expanding().max()
    drawdown_usd = running_max_equity - equity_curve
    max_drawdown_usd = drawdown_usd.max()
    
    # PnL por Trade en USD (considerando interés compuesto)
    capital_before_trade = pd.Series(initial_capital, index=returns.index)
    if len(returns) > 1:
        # El capital antes del trade i es el equity del trade i-1
        capital_before_trade.iloc[1:] = equity_curve.iloc[:-1].values
        
    trade_pnls_usd = returns * capital_before_trade
    avg_trade_usd = trade_pnls_usd.mean()
    
    metrics = {
        'total_return': float(total_return),
        'n_trades': int(n_trades),
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
        'avg_return_per_trade': float(returns.mean()),
        'volatility': float(returns.std()),
        
        # Métricas Nominales
        'net_profit_usd': float(net_profit_usd),
        'final_equity': float(final_equity),
        'max_drawdown_usd': float(max_drawdown_usd),
        'avg_trade_usd': float(avg_trade_usd)
    }
    
    return metrics
