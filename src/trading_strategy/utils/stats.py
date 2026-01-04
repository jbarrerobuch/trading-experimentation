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


def calculate_trade_statistics(
    trades_df: pd.DataFrame, 
    returns_col: str = 'pnl_pct',
    initial_capital: float = 10000.0
) -> Dict[str, float]:
    """
    Calcula estadísticas detalladas de performance a partir de un DataFrame de trades.
    
    Parameters:
    -----------
    trades_df : DataFrame
        DataFrame con columna de retornos por trade
    returns_col : str
        Nombre de la columna de retornos (default: 'pnl_pct')
    initial_capital : float
        Capital inicial para cálculos de equity y drawdown nominal
    
    Returns:
    --------
    dict : Diccionario con estadísticas de trading completas
    """
    if trades_df.empty:
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
            'worst_trade': 0.0
        }

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

    returns = trades_df[returns_col]
    
    # 1. Métricas Básicas
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    total_trades = len(returns)
    n_wins = len(wins)
    n_losses = len(losses)
    
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
    
    # 2. Métricas de Equity y Drawdown (Trade-based)
    # Asumimos reinversión compuesta para la curva de equity
    equity_curve = (1 + returns).cumprod() * initial_capital
    final_equity = equity_curve.iloc[-1]
    net_profit_usd = final_equity - initial_capital
    total_return = (final_equity / initial_capital) - 1
    
    # Drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Drawdown USD
    drawdown_usd = running_max - equity_curve
    max_drawdown_usd = drawdown_usd.max()
    
    # 3. Métricas Avanzadas (Trade-based)
    
    # Sharpe Ratio (Trade-based)
    std_ret = returns.std()
    sharpe_ratio = (avg_trade / std_ret * np.sqrt(total_trades)) if std_ret > 0 else 0.0
    
    # Sortino Ratio (Trade-based)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    # Sortino usa downside deviation de todos los retornos (asumiendo target=0)
    # Una aproximación común es usar std de solo los negativos, pero ajustado al total
    # Aquí usamos la std de los negativos como proxy simple
    sortino_ratio = (avg_trade / downside_std * np.sqrt(total_trades)) if downside_std > 0 else 0.0
    
    # Calmar Ratio
    calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # SQN (System Quality Number)
    sqn = (np.sqrt(total_trades) * (avg_trade / std_ret)) if std_ret > 0 else 0.0
    
    # Kelly Criterion
    if risk_reward > 0:
        kelly = win_rate - (1 - win_rate) / risk_reward
    else:
        kelly = 0.0
        
    # Expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

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
        'worst_trade': float(worst_trade)
    }
