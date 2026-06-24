import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def is_outlier(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando el método IQR (Interquartile Range)
    
    Parameters:
    -----------
    series : pd.Series
        Serie de datos numéricos
    factor : float
        Factor multiplicador del IQR (default: 1.5)
        
    Returns:
    --------
    pd.Series (bool) : True si es outlier
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (series < lower_bound) | (series > upper_bound)


def is_outlier_mod_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Detecta outliers usando Modified Z-Score (robusto a no-normalidad)
    
    Parameters:
    -----------
    series : pd.Series
        Serie de datos numéricos
    threshold : float
        Umbral para considerar outlier (default: 3.5)
        
    Returns:
    --------
    pd.Series (bool) : True si es outlier
    """
    median = series.median()
    mad = (series - median).abs().median()
    
    if mad == 0:
        return pd.Series(False, index=series.index)
    
    mod_z_score = 0.6745 * (series - median) / mad
    return mod_z_score.abs() > threshold


def _col_array(trades_df, name):
    """Devuelve la columna como array float, o None si no existe."""
    if name in trades_df.columns:
        return np.asarray(trades_df[name], dtype=float)
    return None


def calculate_trade_statistics(
    trades_df,
    returns_col: str = 'pnl_pct',
    initial_capital: float = 10000.0,
    costs_pct=None,
    commission_pct=None,
    slippage_pct=None,
) -> Dict[str, float]:
    """
    Calcula estadísticas detalladas de performance a partir de los trades.

    Parameters:
    -----------
    trades_df : DataFrame | np.ndarray | pd.Series
        DataFrame con columna de retornos por trade, o directamente un array de
        retornos NETOS (pnl_pct). El array es la ruta rápida del grid search.
    returns_col : str
        Nombre de la columna de retornos (default: 'pnl_pct'). Solo con DataFrame.
    initial_capital : float
        Capital inicial para cálculos de equity y drawdown nominal.
    costs_pct, commission_pct, slippage_pct : array, optional
        Costes efectivos por trade (fracciones). Si se proveen (o están como columnas
        del DataFrame), se añaden agregados nominales (total_costs_usd, etc.).

    Returns:
    --------
    dict : Diccionario con estadísticas de trading completas
    """
    # Ruta rápida: array/Series de retornos directamente
    if isinstance(trades_df, (np.ndarray, pd.Series)):
        returns = np.asarray(trades_df, dtype=float)
        if returns.size == 0:
            return _empty_stats(initial_capital)
        return _stats_from_returns(returns, initial_capital,
                                   costs_pct, commission_pct, slippage_pct)

    if trades_df.empty:
        return _empty_stats(initial_capital)

    # Normalizar nombre de columna si es necesario
    if returns_col not in trades_df.columns:
        if 'pnl_pct' in trades_df.columns:
            returns_col = 'pnl_pct'
        elif 'profit_pct' in trades_df.columns:
            returns_col = 'profit_pct'
        elif 'returns' in trades_df.columns:
            returns_col = 'returns'

    if returns_col not in trades_df.columns:
        # Si no se encuentra la columna, retornar vacíos
        return {
            'total_trades': len(trades_df),
            'error': 'Returns column not found' # pyright: ignore[reportReturnType]
        }

    returns = np.asarray(trades_df[returns_col], dtype=float)
    # Si el DataFrame trae columnas de costes, usarlas para los agregados nominales
    if costs_pct is None:
        costs_pct = _col_array(trades_df, 'costs_pct')
    if commission_pct is None:
        commission_pct = _col_array(trades_df, 'commission_pct')
    if slippage_pct is None:
        slippage_pct = _col_array(trades_df, 'slippage_pct')
    return _stats_from_returns(returns, initial_capital,
                               costs_pct, commission_pct, slippage_pct)


def _empty_stats(initial_capital: float) -> Dict[str, float]:
    """Métricas por defecto cuando no hay trades."""
    return {
        'total_trades': 0,
        'win_rate': 0.0,
        'total_return': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'calmar_ratio': 0.0,
        'max_drawdown': 0.0,
        'profit_factor': 0.0,
        'avg_trade': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'risk_reward_ratio': 0.0,
        'sqn': 0.0,
        'kelly_criterion': 0.0,
        'expectancy': 0.0,
        'net_profit_usd': 0.0,
        'final_equity': initial_capital,
        'max_drawdown_usd': 0.0,
        'best_trade': 0.0,
        'worst_trade': 0.0,
        'total_costs_pct': 0.0,
        'total_commission_usd': 0.0,
        'total_slippage_usd': 0.0,
        'total_costs_usd': 0.0,
    }


def _stats_from_returns(
    returns: np.ndarray,
    initial_capital: float = 10000.0,
    costs_pct=None,
    commission_pct=None,
    slippage_pct=None,
) -> Dict[str, float]:
    """
    Calcula las estadísticas a partir de un array numpy de retornos NETOS por trade.

    Replica exactamente la semántica de la implementación basada en pandas
    (incluyendo std con ddof=1). Si se proveen los costes por trade, añade los
    agregados nominales usando el notional compuesto (equity antes de cada trade).
    """
    # 1. Métricas Básicas
    wins = returns[returns > 0]
    losses = returns[returns <= 0]          # incluye ceros (igual que el original)

    total_trades = returns.size
    n_wins = wins.size
    n_losses = losses.size

    win_rate = n_wins / total_trades if total_trades > 0 else 0.0
    avg_win = wins.mean() if n_wins > 0 else 0.0
    avg_loss = losses.mean() if n_losses > 0 else 0.0
    avg_trade = returns.mean()

    best_trade = returns.max()
    worst_trade = returns.min()

    # Profit Factor
    gross_profit = wins.sum() if n_wins > 0 else 0.0
    gross_loss = abs(losses.sum()) if n_losses > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Risk Reward
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    # 2. Equity y Drawdown (reinversión compuesta)
    equity_curve = np.cumprod(1.0 + returns) * initial_capital
    final_equity = equity_curve[-1]
    net_profit_usd = final_equity - initial_capital
    total_return = (final_equity / initial_capital) - 1

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    drawdown_usd = running_max - equity_curve
    max_drawdown_usd = drawdown_usd.max()

    # 3. Métricas Avanzadas (pandas usa ddof=1 en std)
    std_ret = returns.std(ddof=1) if total_trades > 1 else np.nan
    sharpe_ratio = (avg_trade / std_ret * np.sqrt(total_trades)) if std_ret > 0 else 0.0

    downside_returns = returns[returns < 0]
    if downside_returns.size == 0:
        downside_std = 0.0
    elif downside_returns.size == 1:
        downside_std = np.nan   # pandas .std(ddof=1) de 1 elemento -> NaN
    else:
        downside_std = downside_returns.std(ddof=1)
    sortino_ratio = (avg_trade / downside_std * np.sqrt(total_trades)) if downside_std > 0 else 0.0

    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    sqn = (np.sqrt(total_trades) * (avg_trade / std_ret)) if std_ret > 0 else 0.0

    kelly = win_rate - (1 - win_rate) / risk_reward if risk_reward > 0 else 0.0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    # 4. Costes efectivos (nominales). Notional por trade = equity ANTES del trade
    #    (reinversión compuesta): equity_before[0]=capital, equity_before[i]=equity tras i-1.
    total_costs_pct = float(np.sum(costs_pct)) if costs_pct is not None else 0.0
    total_commission_usd = 0.0
    total_slippage_usd = 0.0
    total_costs_usd = 0.0
    if commission_pct is not None or slippage_pct is not None or costs_pct is not None:
        equity_before = np.empty_like(returns)
        equity_before[0] = initial_capital
        equity_before[1:] = equity_curve[:-1]
        if commission_pct is not None:
            total_commission_usd = float(np.sum(np.asarray(commission_pct, dtype=float) * equity_before))
        if slippage_pct is not None:
            total_slippage_usd = float(np.sum(np.asarray(slippage_pct, dtype=float) * equity_before))
        if costs_pct is not None:
            total_costs_usd = float(np.sum(np.asarray(costs_pct, dtype=float) * equity_before))

    return {
        'total_trades': int(total_trades),
        'winning_trades': int(n_wins),
        'losing_trades': int(n_losses),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'avg_trade': float(avg_trade),
        'profit_factor': float(profit_factor),
        'risk_reward_ratio': float(risk_reward),
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio),
        'max_drawdown': float(max_drawdown),
        'sqn': float(sqn),
        'kelly_criterion': float(kelly),
        'expectancy': float(expectancy),
        'net_profit_usd': float(net_profit_usd),
        'final_equity': float(final_equity),
        'max_drawdown_usd': float(max_drawdown_usd),
        'best_trade': float(best_trade),
        'worst_trade': float(worst_trade),
        'total_costs_pct': float(total_costs_pct),
        'total_commission_usd': float(total_commission_usd),
        'total_slippage_usd': float(total_slippage_usd),
        'total_costs_usd': float(total_costs_usd),
    }
