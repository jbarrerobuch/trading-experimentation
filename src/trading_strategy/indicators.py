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

# ----------------------------------------------------------------------------
# Cache de valores de indicador (value cache) para grid search.
#
# La parte cara de un experimento es la llamada a pandas-ta (p.ej. CCI con
# period grande). Su resultado depende SOLO de la función y de sus parámetros
# escalares (los datos OHLCV son constantes durante un grid search). Reutilizarlo
# evita recálculos cuando solo varían los umbrales (threshold) o el position_type.
#
# IMPORTANTE: un value_cache está ligado implícitamente a un único dataset (las
# Series OHLCV se excluyen de la clave). Debe crearse uno nuevo por DataFrame.
# ----------------------------------------------------------------------------
_VALUE_CACHE_MAX = 128
_DATA_SENTINEL = object()  # marcador para argumentos de datos (Series/arrays)


def new_value_cache() -> "OrderedDict":
    """Crea un value cache (LRU) para reutilizar durante un grid search."""
    return OrderedDict()


def _scalar_key(value):
    """Reduce un argumento a algo hasheable; los datos se reemplazan por un sentinel."""
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return _DATA_SENTINEL


class _TaProxy:
    """
    Proxy sobre el módulo ``pandas_ta`` que memoiza las llamadas en un value cache.

    Se usa shadowando el nombre ``ta`` dentro de :func:`compute_signal`, de modo
    que cada ``ta.func(...)`` pasa por aquí sin tocar el resto del módulo. La clave
    excluye los argumentos de datos (Series), constantes durante el grid search.
    """
    __slots__ = ('_cache',)

    def __init__(self, cache):
        self._cache = cache

    def __getattr__(self, name):
        func = getattr(ta, name)
        cache = self._cache
        if cache is None:
            return func

        def wrapped(*args, **kwargs):
            key = (name,
                   tuple(_scalar_key(a) for a in args),
                   tuple(sorted((k, _scalar_key(v)) for k, v in kwargs.items())))
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                return hit
            val = func(*args, **kwargs)
            if val is not None:  # no cacheamos fallos (None)
                cache[key] = val
                if len(cache) > _VALUE_CACHE_MAX:
                    cache.popitem(last=False)
            return val

        return wrapped


def calculate_returns_and_momentum(
    df: pd.DataFrame, 
    lookback_periods: List[int] = [10], 
    lookforward_periods: List[int] = [], 
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
    if lookforward_periods:
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


def compute_signal(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    volume_s: Optional[pd.Series],
    indicator: str,
    params: Dict[str, Any],
    want_aux: bool = False,
    value_cache: Optional["OrderedDict"] = None,
) -> tuple:
    """
    Motor de señales (hot path). Calcula UN indicador a partir de las Series OHLCV
    y devuelve únicamente el array de señal (-1/0/1), sin copiar ni mutar ningún
    DataFrame grande.

    Esta es la única fuente de verdad para la lógica de señales. La función pública
    ``calculate_indicator_and_signals`` es un wrapper delgado sobre ésta.

    Parameters
    ----------
    open_s, high_s, low_s, close_s, volume_s : pd.Series
        Series OHLCV (referencias, no se copian).
    indicator : str
        Nombre del indicador.
    params : dict
        Parámetros del indicador.
    want_aux : bool
        Si True, además del array de señal se devuelve un dict con las Series
        auxiliares (indicator_value, macd, stoch_k, ...) que usan los gráficos.
        En el hot path se deja en False para evitar trabajo innecesario.

    Returns
    -------
    (signal : np.ndarray, aux : dict)
        signal es un array float con valores en {-1, 0, 1} (NaN -> 0).
        aux es {} si want_aux es False.
    """
    aux: Dict[str, Any] = {}
    n = len(close_s)
    signal = None  # np.ndarray al final

    # Shadow local de 'ta': memoiza las llamadas a pandas-ta en el value cache.
    # (Si value_cache es None, el proxy delega directamente sin cachear.)
    ta = _TaProxy(value_cache)

    def _aux(name, series):
        if want_aux:
            aux[name] = series

    if indicator == 'rsi':
        period = int(params.get('period', 14))
        overbought = params.get('overbought', 70)
        oversold = params.get('oversold', 30)
        val = ta.rsi(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > overbought, -1, np.where(val < oversold, 1, 0))

    elif indicator == 'macd':
        fast = int(params.get('fast', 12))
        slow = int(params.get('slow', 26))
        signal_period = int(params.get('signal', 9))
        macd_result = ta.macd(close_s, fast=fast, slow=slow, signal=signal_period)
        if macd_result is not None:
            cols = macd_result.columns
            macd_col = next((c for c in cols if c.startswith('MACD_') and not c.startswith('MACDh') and not c.startswith('MACDs')), None)
            signal_col = next((c for c in cols if c.startswith('MACDs_')), None)
            hist_col = next((c for c in cols if c.startswith('MACDh_')), None)
            if macd_col and signal_col:
                macd_line = macd_result[macd_col]
                macd_sig = macd_result[signal_col]
                _aux('macd', macd_line)
                _aux('macd_signal_line', macd_sig)
                _aux('macd_histogram', macd_result[hist_col] if hist_col else (macd_line - macd_sig))
                signal = np.where(macd_line > macd_sig, 1, -1)

    elif indicator == 'willr':
        period = int(params.get('period', 14))
        high_threshold = params.get('high_threshold', -20)
        low_threshold = params.get('low_threshold', -80)
        val = ta.willr(high_s, low_s, close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > high_threshold, -1, np.where(val < low_threshold, 1, 0))

    elif indicator == 'adx':
        period = int(params.get('period', 14))
        threshold = params.get('threshold', 25)
        adx_result = ta.adx(high_s, low_s, close_s, length=period)
        if adx_result is not None:
            adx = adx_result[f'ADX_{period}']
            plus = adx_result[f'DMP_{period}']
            minus = adx_result[f'DMN_{period}']
            _aux('adx', adx); _aux('adx_plus', plus); _aux('adx_minus', minus)
            signal = np.where((adx > threshold) & (plus > minus), 1,
                              np.where((adx > threshold) & (plus < minus), -1, 0))

    elif indicator == 'stoch':
        k = int(params.get('k', 14))
        d = int(params.get('d', 3))
        smooth_k = int(params.get('smooth_k', 3))
        stoch = ta.stoch(high_s, low_s, close_s, k=k, d=d, smooth_k=smooth_k)
        if stoch is not None:
            k_col = next((c for c in stoch.columns if c.startswith('STOCHk_')), None)
            d_col = next((c for c in stoch.columns if c.startswith('STOCHd_')), None)
            if k_col and d_col:
                sk = stoch[k_col]; sd = stoch[d_col]
                _aux('stoch_k', sk); _aux('stoch_d', sd)
                signal = np.where(sk > sd, 1, -1)

    elif indicator == 'cci':
        period = int(params.get('period', 20))
        threshold = float(params.get('threshold', 100))
        val = ta.cci(high_s, low_s, close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > threshold, -1, np.where(val < -threshold, 1, 0))

    elif indicator == 'er':
        period = int(params.get('period', 10))
        threshold = float(params.get('threshold', 0.3))
        val = ta.er(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > threshold, 1, -1)

    elif indicator == 'inertia':
        period = int(params.get('period', 20))
        upper = float(params.get('upper', 60))
        lower = float(params.get('lower', 40))
        val = ta.inertia(close_s, high_s, low_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > upper, 1, np.where(val < lower, -1, 0))

    elif indicator == 'rsx':
        period = int(params.get('period', 14))
        overbought = float(params.get('overbought', 70))
        oversold = float(params.get('oversold', 30))
        val = ta.rsx(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > overbought, -1, np.where(val < oversold, 1, 0))

    elif indicator == 'slope':
        period = int(params.get('period', 10))
        val = ta.slope(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.sign(val.to_numpy())

    elif indicator == 'trix':
        period = int(params.get('period', 14))
        signal_period = int(params.get('signal', 9))
        trix = ta.trix(close_s, length=period, signal=signal_period)
        if trix is not None:
            trix_col = next((c for c in trix.columns if c.startswith('TRIX_') and not c.startswith('TRIXs_')), None)
            if trix_col:
                val = trix[trix_col]
                _aux('indicator_value', val)
                signal = np.where(val > 0, 1, -1)

    elif indicator == 'uo':
        # Ultimate Oscillator (antes mezclado por error dentro del bloque 'trix')
        threshold = float(params.get('threshold', 50))
        val = ta.uo(high_s, low_s, close_s)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > threshold, 1, -1)

    elif indicator == 'bop':
        val = ta.bop(open_s, high_s, low_s, close_s)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > 0, 1, -1)

    elif indicator == 'cmo':
        period = int(params.get('period', 14))
        val = ta.cmo(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > 0, 1, -1)

    elif indicator == 'ao':
        fast = int(params.get('fast', 5))
        slow = int(params.get('slow', 34))
        val = ta.ao(high_s, low_s, fast=fast, slow=slow)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > 0, 1, -1)

    elif indicator == 'mom':
        period = int(params.get('period', 10))
        val = ta.mom(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.sign(val.to_numpy())

    elif indicator == 'roc':
        period = int(params.get('period', 10))
        val = ta.roc(close_s, length=period)
        if val is not None:
            _aux('indicator_value', val)
            signal = np.where(val > 0, 1, -1)

    elif indicator == 'bbp':
        period = int(params.get('period', 20))
        std_dev = float(params.get('std_dev', 2.0))
        lower_thresh = float(params.get('lower_threshold', 0.0))
        upper_thresh = float(params.get('upper_threshold', 1.0))
        bb = ta.bbands(close_s, length=period, std=std_dev)  # pyright: ignore[reportArgumentType]
        if bb is not None:
            bbp_col = next((c for c in bb.columns if c.startswith('BBP_')), None)
            if bbp_col:
                val = bb[bbp_col]
                _aux('indicator_value', val)
                signal = np.where(val > upper_thresh, -1, np.where(val < lower_thresh, 1, 0))

    elif indicator == 'supertrend':
        period = int(params.get('period', 10))
        multiplier = float(params.get('multiplier', 3.0))
        st = ta.supertrend(high_s, low_s, close_s, length=period, multiplier=multiplier)
        if st is not None:
            trend_col = next((c for c in st.columns if c.startswith('SUPERTd_')), None)
            if trend_col:
                val = st[trend_col]
                _aux('indicator_value', val)
                signal = np.where(val > 0, 1, -1)

    elif indicator == 'psar':
        af0 = float(params.get('af0', 0.02))
        af = float(params.get('af', 0.02))
        max_af = float(params.get('max_af', 0.2))
        psar = ta.psar(high_s, low_s, close_s, af0=af0, af=af, max_af=max_af)
        if psar is not None:
            long_col = next((c for c in psar.columns if c.startswith('PSARl_')), None)
            short_col = next((c for c in psar.columns if c.startswith('PSARs_')), None)
            if long_col and short_col:
                psar_val = psar[long_col].fillna(psar[short_col])
                _aux('indicator_value', psar_val)
                signal = np.where(close_s.to_numpy() > psar_val.to_numpy(), 1, -1)

    elif indicator == 'vortex':
        period = int(params.get('period', 14))
        vortex = ta.vortex(high_s, low_s, close_s, length=period)
        if vortex is not None:
            plus_col = next((c for c in vortex.columns if c.startswith('VTXp_')), None)
            minus_col = next((c for c in vortex.columns if c.startswith('VTXm_')), None)
            if plus_col and minus_col:
                plus = vortex[plus_col]; minus = vortex[minus_col]
                _aux('vortex_plus', plus); _aux('vortex_minus', minus)
                _aux('indicator_value', plus - minus)
                signal = np.where(plus > minus, 1, -1)

    elif indicator == 'aroon':
        period = int(params.get('period', 14))
        aroon = ta.aroon(high_s, low_s, length=period)
        if aroon is not None:
            osc_col = next((c for c in aroon.columns if c.startswith('AROONOSC_')), None)
            if osc_col:
                val = aroon[osc_col]
                _aux('indicator_value', val)
                signal = np.where(val > 0, 1, -1)

    elif indicator == 'fisher':
        period = int(params.get('period', 9))
        signal_period = int(params.get('signal', 1))
        fisher = ta.fisher(high_s, low_s, length=period, signal=signal_period)
        if fisher is not None:
            fisher_col = next((c for c in fisher.columns if c.startswith('FISHERT_') and not c.startswith('FISHERTs_')), None)
            signal_col = next((c for c in fisher.columns if c.startswith('FISHERTs_')), None)
            if fisher_col:
                val = fisher[fisher_col]
                _aux('indicator_value', val)
                if signal_col and signal_period > 1:
                    fsig = fisher[signal_col]
                    _aux('fisher_signal', fsig)
                    signal = np.where(val > fsig, 1, -1)
                else:
                    signal = np.where(val > 0, 1, -1)

    elif indicator == 'keltner':
        period = int(params.get('period', 20))
        atr_period = int(params.get('atr_period', 10))
        multiplier = float(params.get('multiplier', 2.0))
        try:
            kc = ta.kc(high_s, low_s, close_s, length=period, atr_length=atr_period, scalar=multiplier)
            if kc is not None and not kc.empty:
                upper = kc.iloc[:, 0]; lower = kc.iloc[:, 1]; basis = kc.iloc[:, 2]
                _aux('kc_upper', upper); _aux('kc_lower', lower); _aux('kc_basis', basis)
                signal = np.where(close_s.to_numpy() > upper.to_numpy(), 1,
                                  np.where(close_s.to_numpy() < lower.to_numpy(), -1, 0))
        except Exception as e:
            print(f"Error calculando Keltner Channels: {e}")

    elif indicator == 'donchian':
        period = int(params.get('period', 20))
        try:
            dc = ta.donchian(high_s, low_s, length=period)
            if dc is not None and not dc.empty:
                upper = dc.iloc[:, 0]; lower = dc.iloc[:, 1]
                mid = dc.iloc[:, 2] if dc.shape[1] > 2 else (upper + lower) / 2
                _aux('dc_upper', upper); _aux('dc_lower', lower); _aux('dc_mid', mid)
                signal = np.where(close_s.to_numpy() > upper.to_numpy(), 1,
                                  np.where(close_s.to_numpy() < lower.to_numpy(), -1, 0))
        except Exception as e:
            print(f"Error calculando Donchian Channels: {e}")

    elif indicator == 'bbands_width':
        period = int(params.get('period', 20))
        std_dev = float(params.get('std_dev', 2.0))
        try:
            bb = ta.bbands(close_s, length=period, std=std_dev)
            if bb is not None and not bb.empty:
                bbl_col = next((c for c in bb.columns if c.startswith('BBL_')), None)
                bbm_col = next((c for c in bb.columns if c.startswith('BBM_')), None)
                bbu_col = next((c for c in bb.columns if c.startswith('BBU_')), None)
                if bbl_col and bbm_col and bbu_col:
                    bb_lower = bb[bbl_col]; bb_middle = bb[bbm_col]; bb_upper = bb[bbu_col]
                    bb_width = bb_upper - bb_lower
                    bb_width_pct = (bb_width / bb_middle.replace(0, np.nan) * 100).fillna(0)
                    bb_expansion = bb_width.pct_change().fillna(0)
                    squeeze_threshold = bb_width.rolling(20).quantile(0.2)
                    _aux('bb_width', bb_width); _aux('bb_width_pct', bb_width_pct); _aux('bb_expansion', bb_expansion)
                    signal = np.where(bb_width < squeeze_threshold.fillna(bb_width), 1,
                                      np.where(bb_expansion > 0, 1, -1))
        except Exception as e:
            print(f"Error calculando BBands Width: {e}")

    elif indicator == 'stoch_rsi':
        rsi_period = int(params.get('rsi_period', 14))
        k_period = int(params.get('k_period', 3))
        d_period = int(params.get('d_period', 3))
        overbought = float(params.get('overbought', 0.8))
        oversold = float(params.get('oversold', 0.2))
        try:
            stoch_rsi = ta.stochrsi(close_s, length=rsi_period, rsi_length=rsi_period, k=k_period, d=d_period)
            if stoch_rsi is not None and not stoch_rsi.empty:
                k_col = next((c for c in stoch_rsi.columns if c.startswith('STOCHRSIk_')), None)
                d_col = next((c for c in stoch_rsi.columns if c.startswith('STOCHRSId_')), None)
                if k_col and d_col:
                    sk = stoch_rsi[k_col]; sd = stoch_rsi[d_col]
                    _aux('stoch_rsi_k', sk); _aux('stoch_rsi_d', sd)
                    signal = np.where(sk > overbought, -1, np.where(sk < oversold, 1, 0)).astype(float)
                    # Override por crossover K-D (posicional)
                    kd_sign = np.sign((sk - sd).fillna(0).to_numpy())
                    kd_cross = np.empty_like(kd_sign)
                    kd_cross[0] = 0
                    kd_cross[1:] = kd_sign[1:] - kd_sign[:-1]
                    cross_idx = np.nonzero(kd_cross != 0)[0]
                    if len(cross_idx) > 0:
                        signal[cross_idx] = np.where(kd_cross[cross_idx] > 0, 1, -1)
        except Exception as e:
            print(f"Error calculando Stochastic RSI: {e}")

    elif indicator == 'mfi':
        period = int(params.get('period', 14))
        overbought = float(params.get('overbought', 80))
        oversold = float(params.get('oversold', 20))
        try:
            mfi = ta.mfi(high_s, low_s, close_s, volume_s, length=period)
            if mfi is not None:
                if isinstance(mfi, pd.DataFrame):
                    mfi_col = next((c for c in mfi.columns if c.startswith('MFI_')), None)
                    mfi = mfi[mfi_col] if mfi_col else mfi.iloc[:, 0]
                _aux('mfi', mfi)
                signal = np.where(mfi > overbought, -1, np.where(mfi < oversold, 1, 0))
        except Exception as e:
            print(f"Error calculando MFI: {e}")

    # Indicador no implementado o cálculo fallido -> señal neutral
    if signal is None:
        signal = np.zeros(n, dtype=float)
    else:
        # Rellenar NaN con 0 (equivalente a df[signal].fillna(0))
        signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)

    return signal, aux


def calculate_indicator_and_signals(
    df: pd.DataFrame,
    indicator: str,
    params: Dict[str, Any],
    inplace: bool = False,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Wrapper de compatibilidad sobre :func:`compute_signal`.

    Calcula un indicador específico y agrega la columna 'signal' (y columnas
    auxiliares como 'indicator_value', 'macd', etc. usadas por los gráficos).

    Parameters
    ----------
    df : DataFrame
        DataFrame con datos OHLCV.
    indicator : str
        Nombre del indicador.
    params : dict
        Parámetros del indicador.
    inplace : bool
        Si False (default) crea una copia antes de añadir columnas.
    use_cache : bool
        Mantenido por compatibilidad de firma; ya no se usa (el cache previo
        no aportaba aciertos en grid search y añadía overhead de copia).

    Returns
    -------
    DataFrame con columna 'signal' (-1, 0, 1) y auxiliares.
    """
    if not inplace:
        df = df.copy()

    volume_s = df[COL_VOLUME] if COL_VOLUME in df.columns else None
    signal, aux = compute_signal(
        df[COL_OPEN], df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], volume_s,
        indicator, params, want_aux=True
    )

    for col, series in aux.items():
        df[col] = series
    df[COL_SIGNAL] = signal
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

    # Series OHLCV (referencias, sin copiar el frame por cada indicador)
    volume_s = df[COL_VOLUME] if COL_VOLUME in df.columns else None
    df[COL_SIGNAL] = compute_combined_signal(
        df[COL_OPEN], df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], volume_s,
        indicators_configs, combination_method
    )
    return df


def compute_combined_signal(
    open_s: pd.Series,
    high_s: pd.Series,
    low_s: pd.Series,
    close_s: pd.Series,
    volume_s: Optional[pd.Series],
    indicators_configs: List[Dict[str, Any]],
    combination_method: str = 'AND',
    value_cache: Optional["OrderedDict"] = None,
) -> np.ndarray:
    """
    Versión "array" de :func:`combine_indicator_signals` para el hot path.

    Calcula la señal de cada indicador con :func:`compute_signal` (sin copiar el
    DataFrame por indicador) y las combina según ``combination_method``,
    devolviendo únicamente el array de señal combinada (-1/0/1).
    """
    signal_arrays = []
    weights = []
    for config in indicators_configs:
        sig, _ = compute_signal(open_s, high_s, low_s, close_s, volume_s,
                                config['indicator'], config['params'],
                                want_aux=False, value_cache=value_cache)
        signal_arrays.append(sig)
        weights.append(config.get('weight', 1.0))

    # Matriz (n_velas x n_indicadores) en numpy para combinar rápido
    signals_values = np.column_stack(signal_arrays)

    if combination_method == 'AND':
        all_ones = (signals_values == 1).all(axis=1)
        all_minus_ones = (signals_values == -1).all(axis=1)
        combined = np.where(all_ones, 1, np.where(all_minus_ones, -1, 0))
    elif combination_method == 'OR':
        any_ones = (signals_values == 1).any(axis=1)
        any_minus_ones = (signals_values == -1).any(axis=1)
        combined = np.where(any_ones, 1, np.where(any_minus_ones, -1, 0))
    elif combination_method == 'MAJORITY':
        count_ones = (signals_values == 1).sum(axis=1)
        count_minus_ones = (signals_values == -1).sum(axis=1)
        threshold = signals_values.shape[1] / 2
        combined = np.where(count_ones > threshold, 1,
                            np.where(count_minus_ones > threshold, -1, 0))
    elif combination_method == 'WEIGHTED':
        weights_array = np.array(weights)
        normalized_sum = signals_values.dot(weights_array) / weights_array.sum()
        combined = np.where(normalized_sum > 0.3, 1, np.where(normalized_sum < -0.3, -1, 0))
    elif combination_method == 'UNANIMOUS_LONG':
        combined = np.where((signals_values == 1).all(axis=1), 1, 0)
    elif combination_method == 'UNANIMOUS_SHORT':
        combined = np.where((signals_values == -1).all(axis=1), -1, 0)
    else:
        combined = np.zeros(signals_values.shape[0], dtype=float)

    return np.nan_to_num(np.asarray(combined, dtype=float), nan=0.0)
