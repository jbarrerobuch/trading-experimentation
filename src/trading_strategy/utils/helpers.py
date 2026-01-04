"""
Utilidades generales
Funciones helper para el framework de trading
"""

import pandas as pd
import numpy as np


def calculate_returns_summary(df, returns_column='returns'):
    """
    Calcula resumen estadístico de retornos
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con columna de retornos
    returns_column : str
        Nombre de la columna de retornos
    
    Returns:
    --------
    dict : Diccionario con estadísticas
    """
    returns = df[returns_column].dropna()
    
    summary = {
        'mean': returns.mean(),
        'std': returns.std(),
        'min': returns.min(),
        'max': returns.max(),
        'skew': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'positive_days': (returns > 0).sum(),
        'negative_days': (returns < 0).sum(),
        'total_days': len(returns)
    }
    
    return summary


def normalize_ohlcv_columns(df):
    """
    Normaliza nombres de columnas OHLCV a formato estándar
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con columnas OHLCV en cualquier formato
    
    Returns:
    --------
    DataFrame con columnas normalizadas
    """
    df = df.copy()
    
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', 'high', 'low', 'close', 'volume']:
            column_mapping[col] = col_lower.capitalize()
    
    df.rename(columns=column_mapping, inplace=True)
    
    return df


def resample_ohlcv(df, timeframe='1D'):
    """
    Resamplea datos OHLCV a un timeframe diferente
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con datos OHLCV (index = timestamp)
    timeframe : str
        Timeframe objetivo ('1h', '4h', '1D', etc.)
    
    Returns:
    --------
    DataFrame resampled
    """
    df = df.copy()
    
    resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled


def format_strategy_config(strategy_name, indicator, params):
    """
    Formatea configuración de estrategia para display
    
    Parameters:
    -----------
    strategy_name : str
        Nombre de la estrategia
    indicator : str
        Nombre del indicador
    params : dict
        Parámetros del indicador
    
    Returns:
    --------
    str : String formateado
    """
    params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
    return f"{strategy_name} ({indicator}: {params_str})"


def generate_run_name(strategy_name, strategy_type, position_type, params=None, indicators_combo=None, combination_method=None):
    """
    Generates a descriptive name for the MLflow run.
    """
    # Clean strategy name
    clean_name = strategy_name.replace('_optimization', '').replace('_strategy', '')
    name_parts = [clean_name]
    
    # Abbreviation map
    abbr_map = {
        'period': 'p', 'length': 'len', 
        'oversold': 'os', 'overbought': 'ob',
        'fast_period': 'fast', 'slow_period': 'slow', 'signal_period': 'sig',
        'std_dev': 'std', 'upper_period': 'up', 'lower_period': 'low'
    }

    def _format_params(p):
        if not p: return ""
        parts = []
        for k in sorted(p.keys()):
            v = p[k]
            key = abbr_map.get(k, k[:3])
            parts.append(f"{key}{v}")
        return "".join(parts)
    
    if strategy_type == 'combo':
        # Add indicators with their params
        if indicators_combo:
            inds_parts = []
            for ind in indicators_combo:
                name = ind.get('indicator', '?')[:3].upper()
                p_str = _format_params(ind.get('params', {}))
                inds_parts.append(f"{name}{p_str}")
            name_parts.append("-".join(inds_parts))
        
        # Add method
        if combination_method:
            name_parts.append(combination_method)
            
    else:
        # Add key params (abbreviated)
        if params:
            name_parts.append(_format_params(params))
            
    # Add position type (L/S/B)
    pos_map = {'long': 'L', 'short': 'S', 'both': 'B'}
    name_parts.append(pos_map.get(position_type, position_type))
    
    return "_".join(name_parts)
