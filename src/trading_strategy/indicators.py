"""
Módulo de indicadores técnicos
Calcula indicadores de momentum, tendencia y volatilidad usando pandas-ta
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from functools import lru_cache
import hashlib
from collections import OrderedDict
from typing import List, Dict, Optional, Union, Any
from .constants import (
    COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME,
    COL_RETURNS, COL_LOG_RETURNS, COL_CANDLE_RTN, COL_FUTURE_RET,
    COL_SIGNAL
)


# Cache global para indicadores calculados con límite LRU
_INDICATOR_CACHE: OrderedDict = OrderedDict()
_CACHE_MAX_SIZE = 256  # Límite de entradas en cache


def calculate_returns_and_momentum(
    df: pd.DataFrame, 
    lookback_periods: List[int] = [5, 10, 20, 30, 60], 
    lookforward_periods: List[int] = [2, 5, 10, 15, 20, 30, 60, 120], 
    compute_indicators: bool = True
) -> pd.DataFrame:
    """
    Calcula retornos y opcionalmente 36 indicadores técnicos usando pandas-ta
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con columnas OHLCV: Open, High, Low, Close, Volume
    lookback_periods : list
        Períodos de lookback para calcular momentum
    lookforward_periods : list
        Períodos hacia el futuro para calcular retornos acumulados (future_ret+N)
    compute_indicators : bool
        Si False, solo calcula retornos básicos (más rápido para grid search)
        Si True, calcula todos los 36 indicadores técnicos
    
    Returns:
    --------
    DataFrame con columnas adicionales de retornos e indicadores
    """
    df = df.copy()
    
    #  ========== RETORNOS BÁSICOS (siempre se calculan) ==========
    df[COL_RETURNS] = df[COL_CLOSE].pct_change()
    df[COL_LOG_RETURNS] = np.log(df[COL_CLOSE] / df[COL_CLOSE].shift(1))
    df[COL_CANDLE_RTN] = (df[COL_CLOSE] - df[COL_OPEN]) / df[COL_OPEN]
    
    # Pre-calcular future_ret UNA VEZ (optimización para backtesting)
    # future_ret+N: Retorno acumulado a N periodos en el futuro
    # future_ret+1 es equivalente al antiguo future_ret
    df[f'{COL_FUTURE_RET}+1'] = df[COL_RETURNS].shift(-1)
    
    # Calcular retornos futuros para diferentes horizontes
    for period in lookforward_periods:
        df[f'{COL_FUTURE_RET}+{period}'] = df[COL_CLOSE].shift(-period) / df[COL_CLOSE] - 1
    
    # Si solo queremos retornos (para grid search), retornar aquí
    if not compute_indicators:
        return df.dropna(subset=[COL_RETURNS, COL_LOG_RETURNS])
    
    # ========== INDICADORES TÉCNICOS (36 indicadores con pandas-ta) ==========
    # La lógica completa se mantiene en el notebook para edición interactiva
    # Aquí incluimos los más importantes para el grid search
    
    # 1-3. Momentum clásico
    for period in [5, 10, 14, 20, 30]:
        if period in lookback_periods or period in [10, 14, 20]:
            df[f'rsi_{period}'] = ta.rsi(df[COL_CLOSE], length=period)
            df[f'rsi_signal_{period}'] = np.where(df[f'rsi_{period}'] > 70, -1,
                                                  np.where(df[f'rsi_{period}'] < 30, 1, 0))
            
            df[f'roc_{period}'] = ta.roc(df[COL_CLOSE], length=period)
            df[f'roc_signal_{period}'] = np.sign(df[f'roc_{period}'])
            
            df[f'mom_{period}'] = ta.mom(df[COL_CLOSE], length=period)
            df[f'mom_signal_{period}'] = np.sign(df[f'mom_{period}'])
    
    # 4. MACD
    macd_result = ta.macd(df[COL_CLOSE], fast=12, slow=26, signal=9)
    if macd_result is not None:
        df['macd'] = macd_result['MACD_12_26_9']
        df['macd_signal_line'] = macd_result['MACDs_12_26_9']
        df['macd_histogram'] = macd_result['MACDh_12_26_9']
        df['macd_signal'] = np.where(df['macd'] > df['macd_signal_line'], 1, -1)
    
    # 5. Stochastic
    stoch = ta.stoch(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], k=14, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        df['stoch_signal'] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
    
    # 6. CCI
    for period in [20, 30]:
        if period in lookback_periods or period == 20:
            df[f'cci_{period}'] = ta.cci(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=period)
            df[f'cci_signal_{period}'] = np.where(df[f'cci_{period}'] > 0, 1, -1)
    
    # 7. Williams %R
    for period in [14, 20, 30]:
        if period in lookback_periods or period in [14, 20]:
            willr = ta.willr(df['High'], df['Low'], df['Close'], length=period)
            if willr is not None:
                df[f'willr_{period}'] = willr
                df[f'willr_signal_{period}'] = np.where(df[f'willr_{period}'] > -20, -1,
                                                        np.where(df[f'willr_{period}'] < -80, 1, 0))
    
    # 17. ADX
    adx_result = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_result is not None:
        df['adx'] = adx_result['ADX_14']
        df['adx_plus'] = adx_result['DMP_14']
        df['adx_minus'] = adx_result['DMN_14']
        df['adx_signal'] = np.where(df['adx_plus'] > df['adx_minus'], 1, -1)
    
    # 18. Efficiency Ratio
    for period in [10, 20]:
        if period in lookback_periods or period == 10:
            er = ta.er(df['Close'], length=period)
            if er is not None:
                df[f'er_{period}'] = er
                df[f'er_signal_{period}'] = np.where(df[f'er_{period}'] > 0.3, 1, -1)
    
    # 20. Inertia
    for period in [14, 20]:
        if period in lookback_periods or period == 14:
            inertia = ta.inertia(df['Close'], df['High'], df['Low'], length=period)
            if inertia is not None:
                df[f'inertia_{period}'] = inertia
                df[f'inertia_signal_{period}'] = np.where(df[f'inertia_{period}'] > 60, 1,
                                                          np.where(df[f'inertia_{period}'] < 40, -1, 0))
    
    # 21. RSX
    for period in [14, 20]:
        if period in lookback_periods or period == 14:
            rsx = ta.rsx(df['Close'], length=period)
            if rsx is not None:
                df[f'rsx_{period}'] = rsx
                df[f'rsx_signal_{period}'] = np.where(df[f'rsx_{period}'] > 70, -1,
                                                      np.where(df[f'rsx_{period}'] < 30, 1, 0))
    
    # 22. Slope
    for period in [10, 20]:
        if period in lookback_periods or period == 10:
            slope = ta.slope(df['Close'], length=period)
            if slope is not None:
                df[f'slope_{period}'] = slope
                df[f'slope_signal_{period}'] = np.sign(df[f'slope_{period}'])
    
    # 25. Trix
    for period in [14, 20]:
        if period in lookback_periods or period == 14:
            trix = ta.trix(df['Close'], length=period, signal=9)
            if trix is not None:
                df[f'trix_{period}'] = trix[f'TRIX_{period}_9']
                df[f'trix_signal_{period}'] = np.where(df[f'trix_{period}'] > 0, 1, -1)
    
    # 26. Ultimate Oscillator
    uo = ta.uo(df['High'], df['Low'], df['Close'])
    if uo is not None:
        df['uo'] = uo
        df['uo_signal'] = np.where(df['uo'] > 50, 1, -1)
    
    # 11. BOP
    bop = ta.bop(df['Open'], df['High'], df['Low'], df['Close'])
    if bop is not None:
        df['bop'] = bop
        df['bop_signal'] = np.where(df['bop'] > 0, 1, -1)
    
    # 14. CMO
    for period in [14, 20]:
        if period in lookback_periods or period == 14:
            cmo = ta.cmo(df['Close'], length=period)
            if cmo is not None:
                df[f'cmo_{period}'] = cmo
                df[f'cmo_signal_{period}'] = np.where(df[f'cmo_{period}'] > 0, 1, -1)
    
    # 8. ATR
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    if atr is not None:
        df['atr_14'] = atr
    
    # 9. AO
    ao = ta.ao(df['High'], df['Low'], fast=5, slow=34)
    if ao is not None:
        df['ao'] = ao
        df['ao_signal'] = np.where(df['ao'] > 0, 1, -1)
    
    # Volatilidad básica
    df['volatility_20'] = df['returns'].rolling(20).std()
    
    # Momentum tradicional
    for period in lookback_periods:
        df[f'momTrad_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'momTrad_signal_{period}'] = np.sign(df[f'momTrad_{period}'])
    
    return df.dropna(subset=['returns', 'log_returns'])


def _get_cache_key(df: pd.DataFrame, indicator: str, params: Dict[str, Any]) -> str:
    """
    Genera una clave única para el cache basada en el DataFrame y parámetros
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con datos
    indicator : str
        Nombre del indicador
    params : dict
        Parámetros del indicador
    
    Returns:
    --------
    str : Clave única para cache
    """
    # Usar hash del DataFrame (basado en index y shape) + parámetros
    df_hash = hashlib.md5(f"{len(df)}_{df.index[0]}_{df.index[-1]}".encode()).hexdigest()[:8]
    params_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{df_hash}_{indicator}_{params_str}"


def clear_indicator_cache() -> None:
    """Limpia el cache de indicadores (útil entre diferentes tickers/timeframes)"""
    global _INDICATOR_CACHE
    _INDICATOR_CACHE.clear()


def calculate_indicator_and_signals(
    df: pd.DataFrame, 
    indicator: str, 
    params: Dict[str, Any], 
    inplace: bool = False, 
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Calcula un indicador específico con parámetros personalizados y genera señales
    
    Usa el DataFrame original y calcula el indicador dinámicamente
    No requiere pre-cálculo de indicadores
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con datos OHLCV
    indicator : str
        Nombre del indicador: 'rsi', 'macd', 'willr', 'stoch', 'cci', 'rsx',
        'cmo', 'uo', 'adx', 'er', 'slope', 'trix', 'ao', 'inertia', 'bop'
    params : dict
        Parámetros del indicador (ej: {'period': 14, 'overbought': 70, 'oversold': 30})
    inplace : bool
        Si True, modifica el DataFrame original (más rápido)
        Si False, crea una copia (más seguro, default)
        Default: False
    use_cache : bool
        Si True, usa cache para indicadores ya calculados (más rápido)
        Default: True
    
    Returns:
    --------
    DataFrame completo con columna 'signal' agregada (-1, 0, 1)
    """
    if not inplace:
        df = df.copy()
    
    # Columnas ya están normalizadas por calculate_returns_and_momentum()
    # No necesitamos normalizar de nuevo (optimización)
    
    # Verificar cache antes de calcular
    cache_key = None
    if use_cache:
        cache_key = _get_cache_key(df, indicator, params)
        if cache_key in _INDICATOR_CACHE:
            # Cache hit! Reutilizar valores calculados
            cached_data = _INDICATOR_CACHE[cache_key]
            for col, values in cached_data.items():
                df[col] = values
            return df
    
    # IMPLEMENTACIÓN PARA CADA INDICADOR
    # (El código completo se mantiene en el notebook)
    # Aquí una versión simplificada
    
    # Diccionario para almacenar columnas calculadas (para cache)
    calculated_columns = {}
    
    if indicator == 'rsi':
        period = params.get('period', 14)
        overbought = params.get('overbought', 70)
        oversold = params.get('oversold', 30)
        
        df['indicator_value'] = ta.rsi(df[COL_CLOSE], length=period)
        df[COL_SIGNAL] = np.where(df['indicator_value'] > overbought, -1,
                               np.where(df['indicator_value'] < oversold, 1, 0))
        
        # Guardar en cache
        if use_cache:
            calculated_columns['indicator_value'] = df['indicator_value'].copy()
            calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
    
    elif indicator == 'macd':
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal_period = params.get('signal', 9)
        
        macd_result = ta.macd(df[COL_CLOSE], fast=fast, slow=slow, signal=signal_period)
        if macd_result is not None:
            # Buscar columnas dinámicamente para evitar errores de nombre
            cols = macd_result.columns
            macd_col = next((c for c in cols if c.startswith('MACD_') and not c.startswith('MACDh') and not c.startswith('MACDs')), None)
            signal_col = next((c for c in cols if c.startswith('MACDs_')), None)

            if macd_col and signal_col:
                df['macd'] = macd_result[macd_col]
                df['macd_signal_line'] = macd_result[signal_col]
                df[COL_SIGNAL] = np.where(df['macd'] > df['macd_signal_line'], 1, -1)
                
                # Guardar en cache
                if use_cache:
                    calculated_columns['macd'] = df['macd'].copy()
                    calculated_columns['macd_signal_line'] = df['macd_signal_line'].copy()
                    calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
            else:
                df[COL_SIGNAL] = 0
        else:
            df[COL_SIGNAL] = 0
    
    elif indicator == 'willr':
        period = params.get('period', 14)
        high_threshold = params.get('high_threshold', -20)
        low_threshold = params.get('low_threshold', -80)
        
        willr = ta.willr(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=period)
        if willr is not None:
            df['indicator_value'] = willr
            df[COL_SIGNAL] = np.where(df['indicator_value'] > high_threshold, -1,
                                   np.where(df['indicator_value'] < low_threshold, 1, 0))
            
            # Guardar en cache
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0
    
    elif indicator == 'adx':
        period = params.get('period', 14)
        threshold = params.get('threshold', 25)
        
        adx_result = ta.adx(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=period)
        if adx_result is not None:
            df['adx'] = adx_result[f'ADX_{period}']
            df['adx_plus'] = adx_result[f'DMP_{period}']
            df['adx_minus'] = adx_result[f'DMN_{period}']
            # Signal solo cuando ADX > threshold (tendencia fuerte)
            df[COL_SIGNAL] = np.where((df['adx'] > threshold) & (df['adx_plus'] > df['adx_minus']), 1,
                                   np.where((df['adx'] > threshold) & (df['adx_plus'] < df['adx_minus']), -1, 0))
            
            # Guardar en cache
            if use_cache:
                calculated_columns['adx'] = df['adx'].copy()
                calculated_columns['adx_plus'] = df['adx_plus'].copy()
                calculated_columns['adx_minus'] = df['adx_minus'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0
    
    elif indicator == 'stoch':
        k = params.get('k', 14)
        d = params.get('d', 3)
        smooth_k = params.get('smooth_k', 3)
        
        stoch = ta.stoch(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], k=k, d=d, smooth_k=smooth_k)
        if stoch is not None:
            # pandas-ta returns columns like STOCHk_14_3_3, STOCHd_14_3_3
            # We need to find them dynamically
            k_col = next((c for c in stoch.columns if c.startswith('STOCHk_')), None)
            d_col = next((c for c in stoch.columns if c.startswith('STOCHd_')), None)
            
            if k_col and d_col:
                df['stoch_k'] = stoch[k_col]
                df['stoch_d'] = stoch[d_col]
                df[COL_SIGNAL] = np.where(df['stoch_k'] > df['stoch_d'], 1, -1)
                
                if use_cache:
                    calculated_columns['stoch_k'] = df['stoch_k'].copy()
                    calculated_columns['stoch_d'] = df['stoch_d'].copy()
                    calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
            else:
                df[COL_SIGNAL] = 0
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'cci':
        period = params.get('period', 20)
        cci = ta.cci(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=period)
        if cci is not None:
            df['indicator_value'] = cci
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 100, -1,
                                   np.where(df['indicator_value'] < -100, 1, 0))
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'er':
        period = params.get('period', 10)
        er = ta.er(df[COL_CLOSE], length=period)
        if er is not None:
            df['indicator_value'] = er
            # ER is efficiency ratio, usually used for trend strength. 
            # Signal logic from calculate_returns_and_momentum: > 0.3 -> 1, else -1
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 0.3, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'inertia':
        period = params.get('period', 20)
        inertia = ta.inertia(df[COL_CLOSE], df[COL_HIGH], df[COL_LOW], length=period)
        if inertia is not None:
            df['indicator_value'] = inertia
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 60, 1,
                                   np.where(df['indicator_value'] < 40, -1, 0))
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'rsx':
        period = params.get('period', 14)
        rsx = ta.rsx(df[COL_CLOSE], length=period)
        if rsx is not None:
            df['indicator_value'] = rsx
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 70, -1,
                                   np.where(df['indicator_value'] < 30, 1, 0))
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'slope':
        period = params.get('period', 10)
        slope = ta.slope(df[COL_CLOSE], length=period)
        if slope is not None:
            df['indicator_value'] = slope
            df[COL_SIGNAL] = np.sign(df['indicator_value'])
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'trix':
        period = params.get('period', 14)
        signal_period = params.get('signal', 9)
        trix = ta.trix(df[COL_CLOSE], length=period, signal=signal_period)
        if trix is not None:
            # TRIX_14_9
            trix_col = next((c for c in trix.columns if c.startswith('TRIX_') and not c.startswith('TRIXs_')), None)
            if trix_col:
                df['indicator_value'] = trix[trix_col]
                df[COL_SIGNAL] = np.where(df['indicator_value'] > 0, 1, -1)
                if use_cache:
                    calculated_columns['indicator_value'] = df['indicator_value'].copy()
                    calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
            else:
                df[COL_SIGNAL] = 0
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'uo':
        uo = ta.uo(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE])
        if uo is not None:
            df['indicator_value'] = uo
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 50, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'bop':
        bop = ta.bop(df[COL_OPEN], df[COL_HIGH], df[COL_LOW], df[COL_CLOSE])
        if bop is not None:
            df['indicator_value'] = bop
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 0, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'cmo':
        period = params.get('period', 14)
        cmo = ta.cmo(df[COL_CLOSE], length=period)
        if cmo is not None:
            df['indicator_value'] = cmo
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 0, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'ao':
        fast = params.get('fast', 5)
        slow = params.get('slow', 34)
        ao = ta.ao(df[COL_HIGH], df[COL_LOW], fast=fast, slow=slow)
        if ao is not None:
            df['indicator_value'] = ao
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 0, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'mom':
        period = params.get('period', 10)
        mom = ta.mom(df[COL_CLOSE], length=period)
        if mom is not None:
            df['indicator_value'] = mom
            df[COL_SIGNAL] = np.sign(df['indicator_value'])
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    elif indicator == 'roc':
        period = params.get('period', 10)
        roc = ta.roc(df[COL_CLOSE], length=period)
        if roc is not None:
            df['indicator_value'] = roc
            df[COL_SIGNAL] = np.where(df['indicator_value'] > 0, 1, -1)
            if use_cache:
                calculated_columns['indicator_value'] = df['indicator_value'].copy()
                calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        else:
            df[COL_SIGNAL] = 0

    else:
        # Indicador no implementado, retornar señal neutral
        df[COL_SIGNAL] = 0
    
    # Rellenar NaN con 0 en la columna signal
    df[COL_SIGNAL] = df[COL_SIGNAL].fillna(0)
    
    # Guardar en cache si hay datos calculados
    if use_cache and cache_key and calculated_columns:
        # Agregar signal al cache si no se agregó antes
        if COL_SIGNAL not in calculated_columns:
            calculated_columns[COL_SIGNAL] = df[COL_SIGNAL].copy()
        
        # Implementar LRU: eliminar entrada más antigua si se alcanza el límite
        if len(_INDICATOR_CACHE) >= _CACHE_MAX_SIZE:
            _INDICATOR_CACHE.popitem(last=False)  # Eliminar primera entrada (más antigua)
        
        _INDICATOR_CACHE[cache_key] = calculated_columns
        _INDICATOR_CACHE.move_to_end(cache_key)  # Marcar como recientemente usado
    
    return df


def combine_indicator_signals(
    df: pd.DataFrame, 
    indicators_configs: List[Dict[str, Any]], 
    combination_method: str = 'AND', 
    inplace: bool = False
) -> pd.DataFrame:
    """
    Combina señales de múltiples indicadores usando diferentes métodos
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame base con datos OHLCV
    indicators_configs : list of dict
        Lista de configuraciones de indicadores
        [
            {'indicator': 'rsi', 'params': {'period': 14, ...}, 'weight': 1.0},
            {'indicator': 'macd', 'params': {'fast': 12, ...}, 'weight': 2.0}
        ]
    combination_method : str
        Método de combinación:
        - 'AND': Todos los indicadores deben estar de acuerdo
        - 'OR': Al menos un indicador da señal
        - 'MAJORITY': Mayoría simple (>50%)
        - 'WEIGHTED': Promedio ponderado por pesos
        - 'UNANIMOUS_LONG': Todos =1 para long, resto 0
        - 'UNANIMOUS_SHORT': Todos =-1 para short, resto 0
    inplace : bool
        Si True, modifica el DataFrame original (más rápido)
        Si False, crea una copia (más seguro, default)
        Default: False
    
    Returns:
    --------
    DataFrame completo con columna 'signal' combinada
    """
    if not inplace:
        df = df.copy()
    
    # Calcular señales de cada indicador
    signals_list = []
    weights = []
    
    for config in indicators_configs:
        indicator_name = config['indicator']
        params = config['params']
        weight = config.get('weight', 1.0)
        
        # Calcular indicador y señal (ahora retorna el df completo)
        # use_cache=True para reutilizar cálculos en estrategias combo
        signal_df = calculate_indicator_and_signals(df, indicator_name, params, inplace=False, use_cache=True)
        signals_list.append(signal_df[COL_SIGNAL])
        weights.append(weight)
    
    # Combinar señales según método
    signals_matrix = pd.concat(signals_list, axis=1)
    # Convertir a numpy para operaciones vectorizadas rápidas
    signals_values = signals_matrix.values
    
    if combination_method == 'AND':
        # Todos deben estar de acuerdo
        # Si todos son 1 -> 1
        # Si todos son -1 -> -1
        # Sino 0
        all_ones = (signals_values == 1).all(axis=1)
        all_minus_ones = (signals_values == -1).all(axis=1)
        df[COL_SIGNAL] = np.where(all_ones, 1, np.where(all_minus_ones, -1, 0))
    
    elif combination_method == 'OR':
        # Al menos uno da señal
        # Prioridad a 1 si hay conflicto (o según lógica original)
        # Original: 1 if any 1, else -1 if any -1, else 0
        any_ones = (signals_values == 1).any(axis=1)
        any_minus_ones = (signals_values == -1).any(axis=1)
        df[COL_SIGNAL] = np.where(any_ones, 1, np.where(any_minus_ones, -1, 0))
    
    elif combination_method == 'MAJORITY':
        # Mayoría simple
        count_ones = (signals_values == 1).sum(axis=1)
        count_minus_ones = (signals_values == -1).sum(axis=1)
        threshold = signals_values.shape[1] / 2
        df[COL_SIGNAL] = np.where(count_ones > threshold, 1, 
                               np.where(count_minus_ones > threshold, -1, 0))
    
    elif combination_method == 'WEIGHTED':
        # Promedio ponderado
        weights_array = np.array(weights)
        # Dot product vectorizado
        weighted_sum = signals_values.dot(weights_array)
        normalized_sum = weighted_sum / weights_array.sum()
        
        # Convertir a señales discretas
        df[COL_SIGNAL] = np.where(normalized_sum > 0.3, 1, 
                               np.where(normalized_sum < -0.3, -1, 0))
    
    elif combination_method == 'UNANIMOUS_LONG':
        # Long solo si todos = 1
        all_ones = (signals_values == 1).all(axis=1)
        df[COL_SIGNAL] = np.where(all_ones, 1, 0)
    
    elif combination_method == 'UNANIMOUS_SHORT':
        # Short solo si todos = -1
        all_minus_ones = (signals_values == -1).all(axis=1)
        df[COL_SIGNAL] = np.where(all_minus_ones, -1, 0)
    
    # Rellenar NaN con 0 en la columna signal
    df[COL_SIGNAL] = df[COL_SIGNAL].fillna(0)
    
    return df
