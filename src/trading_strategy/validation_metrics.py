"""
Métricas de validación anti-sobreajuste para el grid search.

Implementa las carencias/correcciones acordadas en la revisión del prompt de
análisis (v2):

  [C1]  Split train/validación + sub-ventanas DENTRO del backtest
        → cada run registra sharpe_train, sharpe_val, sharpe_val_w1/w2/w3
  [C2]  Corrección por múltiples comparaciones: Deflated Sharpe Ratio (DSR)
        y t-stat con Bonferroni como alternativa simple
  [C3]  Sensibilidad de parámetros (meseta vs pico)
  [C4]  Information ratio vs benchmark por sub-período (IR 2-de-3)
  [C5]  Correlación de señales entre estrategias (redundancia del top)
  [C6]  Sharpe por régimen con criterio EXTERNO (drawdown + rango, no SMA200)
  [C7]  Comisión y slippage separados y parametrizables
  [S1]  Ranking stability ASIMÉTRICA (mejora ≠ degradación)
  [S2]  Filtro de frecuencia a nivel PORTFOLIO, no por estrategia
  [S3]  Bootstrap por bloques → intervalo de confianza del Sharpe
  [S4]  Restricción de capital (nocional mínimo Binance, cuenta 100€)
  [S5]  Deduplicación de runs (export Parquet)

Integración en el grid search (hot path):
    ctx = build_validation_context(df, timeframe, commission, slippage)   # 1 vez
    metrics = compute_validation_metrics(pos, ctx)                        # por run

`compute_validation_metrics` trabaja 100% en numpy sobre arrays precomputados
en el `ValidationContext` (retornos, máscaras train/val, régimen, B&H), de modo
que el coste por run es marginal frente al cálculo del indicador.

Semántica (deliberadamente distinta del motor por-trades de backtesting.py):
retornos close-to-close por período con `pos_exec = shift(pos, signal_delay)`
y coste por turnover `|Δpos| × (commission + slippage)`. `signal_delay=1` es el
estándar sin lookahead; `signal_delay=2` es el test de lookahead (Bloque 5.3).

Sin dependencias nuevas: numpy, pandas, scipy (ya en requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .constants import COL_CLOSE

ANNUAL = 365  # períodos/año para datos diarios; cripto opera todos los días

# Períodos por año según timeframe (cripto: 24/7)
_PERIODS_PER_YEAR = {
    '15m': 4 * 24 * 365,
    '30m': 2 * 24 * 365,
    '1h': 24 * 365,
    '4h': 6 * 365,
    '1d': 365,
}


def periods_per_year_from_timeframe(timeframe: str) -> int:
    """Períodos por año para un timeframe (cripto, mercado 24/7)."""
    tf = str(timeframe).lower()
    try:
        return _PERIODS_PER_YEAR[tf]
    except KeyError:
        raise KeyError(
            f"Timeframe desconocido: {timeframe!r}. "
            f"Soportados: {sorted(_PERIODS_PER_YEAR)}"
        ) from None


# ---------------------------------------------------------------------------
# Utilidades base
# ---------------------------------------------------------------------------

def sharpe(returns: pd.Series, periods_per_year: int = ANNUAL) -> float:
    """Sharpe anualizado sobre retornos por período. NaN-safe."""
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def _sharpe_np(r: np.ndarray, periods_per_year: int) -> float:
    """Sharpe anualizado sobre un array sin NaN (hot path).

    Varianza en un solo paso vía np.dot (BLAS): con retornos centrados cerca
    de cero la cancelación numérica es despreciable y es ~5x más rápido que
    ndarray.std para los tamaños del grid.
    """
    n = r.shape[0]
    if n < 2:
        return np.nan
    m = r.sum() / n
    var = (float(np.dot(r, r)) - n * m * m) / (n - 1)
    if var <= 0:
        return np.nan
    return float(m / np.sqrt(var) * np.sqrt(periods_per_year))


def split_train_val(index, val_frac: float = 0.3):
    """[C1] Corte cronológico. Devuelve (mask_train, mask_val) booleanos.
    El corte es por posición temporal, nunca aleatorio."""
    n = len(index)
    cut = int(np.floor(n * (1.0 - val_frac)))
    mask_train = np.zeros(n, dtype=bool)
    mask_train[:cut] = True
    return mask_train, ~mask_train


def subwindow_sharpes(returns: pd.Series, n_windows: int = 3,
                      periods_per_year: int = ANNUAL) -> list[float]:
    """[C1] Sharpe en n sub-ventanas cronológicas iguales del período dado."""
    r = pd.Series(returns).dropna()
    if len(r) < n_windows * 30:  # mínimo 30 obs por ventana
        return [np.nan] * n_windows
    chunks = np.array_split(r, n_windows)
    return [sharpe(c, periods_per_year) for c in chunks]


def _subwindow_bounds(n: int, n_windows: int = 3) -> np.ndarray:
    """Offsets [0, b1, b2, n] de las sub-ventanas (mismo reparto que
    np.array_split: las primeras ventanas absorben el resto)."""
    base, extra = divmod(n, n_windows)
    sizes = [base + 1 if i < extra else base for i in range(n_windows)]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)


# ---------------------------------------------------------------------------
# [S1] Ranking stability asimétrica
# ---------------------------------------------------------------------------

def asymmetric_stability(rank_train: int, rank_val: int, n_strategies: int,
                         sharpe_val: float, sharpe_floor: float = 0.5) -> float:
    """
    Estabilidad de ranking que NO penaliza la mejora train→val.

    Lógica:
      - Degradación (rank_val > rank_train): penaliza proporcionalmente.
        MACD (1º→10º de 15) → stability baja. Correcto.
      - Mejora o mantenimiento (rank_val <= rank_train): stability = 1.
        Fees momentum (9º→1º) → stability 1. Correcto: mejorar fuera de
        muestra es evidencia A FAVOR, nunca en contra.
      - Puerta de calidad: si sharpe_val < sharpe_floor devuelve 0.
        La consistencia en la mediocridad (14º→14º) no es señal.

    Rango: [0, 1]. Ranks son 1-indexed (1 = mejor).
    """
    if np.isnan(sharpe_val) or sharpe_val < sharpe_floor:
        return 0.0
    if n_strategies <= 1:
        return np.nan
    degradation = max(0, rank_val - rank_train)
    return float(max(0.0, 1.0 - degradation / (n_strategies - 1)))


# ---------------------------------------------------------------------------
# [C2] Múltiples comparaciones: DSR y Bonferroni operacionalizados
# ---------------------------------------------------------------------------

def probabilistic_sharpe_ratio(sr_hat: float, sr_benchmark: float, T: int,
                               skew: float, kurt: float) -> float:
    """PSR de Bailey & López de Prado. sr_hat y sr_benchmark POR PERÍODO
    (no anualizados). kurt = curtosis no-exceso (normal = 3). T = n obs."""
    if T < 3 or np.isnan(sr_hat):
        return np.nan
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2))
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr_trials: float) -> float:
    """SR0: máximo Sharpe esperado bajo H0 (sin señal) tras n_trials pruebas.
    Fórmula de valores extremos con constante de Euler-Mascheroni."""
    if n_trials < 2 or var_sr_trials <= 0:
        return 0.0
    gamma = 0.5772156649
    e = np.e
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_sr_trials) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          var_sr_trials: float) -> dict:
    """
    DSR: probabilidad de que el Sharpe observado sea real DADO que se probaron
    n_trials configuraciones.

    Parámetros:
      returns        retornos por período de la estrategia (validación)
      n_trials       nº total de combinaciones del grid (tamaño real del grid,
                     no solo las reportadas — `n_experiments_enumerated` en el
                     manifest de cada sesión)
      var_sr_trials  varianza de los Sharpe POR PERÍODO entre todos los trials
                     (columna `sr_val_period` del export Parquet)

    Interpretación: DSR > 0.95 → señal defendible tras corrección.
    """
    r = pd.Series(returns).dropna()
    T = len(r)
    if T < 30:
        return {"dsr": np.nan, "sr_period": np.nan, "sr0": np.nan}
    sr_period = float(r.mean() / r.std(ddof=1))
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_sr_trials)
    dsr = probabilistic_sharpe_ratio(sr_period, sr0, T, sk, ku)
    return {"dsr": dsr, "sr_period": sr_period, "sr0": sr0,
            "skew": sk, "kurtosis": ku, "T": T}


def sharpe_tstat_bonferroni(sharpe_annual: float, years: float,
                            n_trials: int) -> dict:
    """Alternativa simple al DSR: t ≈ SR_anual × √años; p-valor bilateral
    corregido por Bonferroni. Útil como sanity check rápido."""
    if np.isnan(sharpe_annual) or np.isnan(years) or years <= 0:
        return {"t": np.nan, "p_raw": np.nan, "p_bonferroni": np.nan,
                "significant": False}
    t = sharpe_annual * np.sqrt(years)
    p_raw = 2.0 * (1.0 - stats.norm.cdf(abs(t)))
    p_adj = min(1.0, p_raw * n_trials)
    return {"t": float(t), "p_raw": float(p_raw), "p_bonferroni": float(p_adj),
            "significant": bool(p_adj < 0.05)}


# ---------------------------------------------------------------------------
# [S3] Bootstrap por bloques → IC del Sharpe
# ---------------------------------------------------------------------------

def block_bootstrap_sharpe_ci(returns: pd.Series, block_size: int = 20,
                              n_boot: int = 2000, ci: float = 0.90,
                              periods_per_year: int = ANNUAL,
                              seed: int = 42) -> dict:
    """
    IC del Sharpe anualizado por bootstrap de bloques circulares (preserva
    autocorrelación local). Dos estrategias cuyos IC se solapan NO son
    distinguibles — no concluyas que una 'gana' a la otra.
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < block_size * 3:
        return {"sharpe": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_size))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_size)[None, :]) % n
        sample = r[idx.ravel()][:n]
        s = sample.std(ddof=1)
        boot[i] = np.nan if s == 0 else sample.mean() / s * np.sqrt(periods_per_year)
    alpha = (1.0 - ci) / 2.0
    return {"sharpe": sharpe(pd.Series(r), periods_per_year),
            "ci_low": float(np.nanpercentile(boot, 100 * alpha)),
            "ci_high": float(np.nanpercentile(boot, 100 * (1 - alpha)))}


# ---------------------------------------------------------------------------
# [C6] Régimen con criterio EXTERNO (no circular con las estrategias)
# ---------------------------------------------------------------------------

def classify_regime(close: pd.Series, dd_bear: float = 0.20,
                    range_flat: float = 0.10, lookback: int = 60,
                    max_window: int = 200) -> pd.Series:
    """
    Clasifica cada barra en {bull, bear, flat} SIN usar SMA200 ni ningún
    indicador testeado como estrategia:
      bear : precio ≥ dd_bear por debajo del máximo móvil de `max_window` barras
      flat : rango (max-min)/min de las últimas `lookback` barras < range_flat
      bull : resto

    `lookback` y `max_window` están en BARRAS del timeframe: para datos no
    diarios escálalos (p.ej. 1h → lookback=60*24, max_window=200*24), como hace
    `build_validation_context`.
    """
    close = pd.Series(close).astype(float)
    roll_max = close.rolling(max_window, min_periods=max(2, max_window // 4)).max()
    drawdown = 1.0 - close / roll_max
    hi = close.rolling(lookback, min_periods=lookback).max()
    lo = close.rolling(lookback, min_periods=lookback).min()
    rng = (hi - lo) / lo
    regime = pd.Series("bull", index=close.index)
    regime[rng < range_flat] = "flat"
    regime[drawdown >= dd_bear] = "bear"  # bear domina sobre flat
    return regime


def sharpe_by_regime(strategy_returns: pd.Series, regime: pd.Series,
                     periods_per_year: int = ANNUAL) -> dict:
    """Sharpe condicional por régimen + selector min(bull, bear).
    El selector primario del análisis es min_regime_sharpe, no el promedio."""
    out = {}
    for reg in ("bull", "bear", "flat"):
        mask = regime.reindex(strategy_returns.index) == reg
        out[f"sharpe_{reg}"] = sharpe(strategy_returns[mask], periods_per_year)
    vals = [out["sharpe_bull"], out["sharpe_bear"]]
    vals = [v for v in vals if not np.isnan(v)]
    out["min_regime_sharpe"] = min(vals) if vals else np.nan
    return out


# ---------------------------------------------------------------------------
# [C4] Information ratio vs benchmark por sub-período
# ---------------------------------------------------------------------------

def information_ratio_by_subperiod(strategy_returns: pd.Series,
                                   benchmark_returns: pd.Series,
                                   n_windows: int = 3,
                                   periods_per_year: int = ANNUAL) -> dict:
    """IR = sharpe(estrategia − benchmark) por sub-ventana. Criterio:
    la estrategia debe batir al benchmark en ≥ 2 de 3 sub-ventanas."""
    active = (pd.Series(strategy_returns) - pd.Series(benchmark_returns)).dropna()
    irs = subwindow_sharpes(active, n_windows, periods_per_year)
    wins = sum(1 for x in irs if not np.isnan(x) and x > 0)
    return {"ir_windows": irs, "windows_beating_bh": wins,
            "passes_2of3": wins >= 2}


# ---------------------------------------------------------------------------
# [C3] Sensibilidad de parámetros (meseta vs pico)
# ---------------------------------------------------------------------------

def param_sensitivity(results: pd.DataFrame, param_col: str,
                      metric_col: str = "sharpe_val",
                      neighborhood: int = 1) -> pd.DataFrame:
    """
    Para cada valor de un parámetro, coeficiente de variación del métrico en
    su vecindario (±neighborhood posiciones en la rejilla ordenada).
      sensitivity bajo  → meseta → parámetro robusto
      sensitivity alto  → pico   → sobreajuste de parámetro
    Devuelve DataFrame con [param, metric_mean, sensitivity].
    """
    g = (results.groupby(param_col)[metric_col].mean()
                .sort_index().dropna())
    vals = g.values
    out = []
    for i, p in enumerate(g.index):
        lo, hi = max(0, i - neighborhood), min(len(vals), i + neighborhood + 1)
        neigh = vals[lo:hi]
        m = np.mean(neigh)
        sens = np.nan if m == 0 else float(np.std(neigh) / abs(m))
        out.append({param_col: p, "metric_mean": float(vals[i]),
                    "sensitivity": sens})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# [C5] Correlación de señales (redundancia del top)
# ---------------------------------------------------------------------------

def signal_correlation_matrix(signals: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlación entre series de señal (0/1) de varias estrategias.
    signals = {nombre: serie_de_posicion}. Correlación > 0,5 entre dos
    estrategias del top = redundancia, no diversificación."""
    df = pd.DataFrame(signals).dropna()
    return df.corr()


# ---------------------------------------------------------------------------
# [S2] + [S4] Frecuencia a nivel portfolio y restricción de capital
# ---------------------------------------------------------------------------

def portfolio_trade_frequency(trades_per_year_by_asset: dict[str, float]) -> dict:
    """El objetivo 15-60 ops/año se evalúa AQUÍ, sumando activos.
    A nivel de run individual solo se exige mínimo estadístico
    (ver min_trades_statistical)."""
    total = float(np.nansum(list(trades_per_year_by_asset.values())))
    return {"portfolio_trades_year": total,
            "in_target_15_60": bool(15 <= total <= 60),
            "by_asset": trades_per_year_by_asset}


def min_trades_statistical(n_trades_total: int, floor: int = 15) -> bool:
    """Filtro de run individual: ≥ floor trades EN TODO EL PERÍODO
    (no por año). Sustituye al antiguo 'n_trades < 10/año'."""
    return n_trades_total >= floor


def capital_feasibility(capital_eur: float, n_assets: int,
                        min_notional_usdt: float = 5.0,
                        eurusdt: float = 1.08,
                        fee_rate: float = 0.001,
                        spread_est: float = 0.001) -> dict:
    """
    [S4] ¿Es operable la cartera con este capital?
      - posición por activo vs nocional mínimo de Binance (~5 USDT)
      - coste de fricción relativo por round-trip (fee×2 + spread)
    Recomienda el nº máximo de activos con margen ≥ 2× sobre el mínimo.
    """
    per_asset = capital_eur * eurusdt / n_assets
    friction_rt = 2 * fee_rate + spread_est
    max_assets = int((capital_eur * eurusdt) // (min_notional_usdt * 2))
    return {"position_usdt": round(per_asset, 2),
            "min_notional_usdt": min_notional_usdt,
            "feasible": per_asset >= min_notional_usdt,
            "margin_over_min": round(per_asset / min_notional_usdt, 2),
            "friction_roundtrip_pct": round(friction_rt * 100, 3),
            "recommended_max_assets": max(1, max_assets)}


# ---------------------------------------------------------------------------
# [S5] Deduplicación del export Parquet
# ---------------------------------------------------------------------------

def dedupe_runs(runs: pd.DataFrame,
                key_cols: tuple[str, ...] = ('experiment_id', 'ticker', 'timeframe'),
                time_col: str = 'session_id') -> pd.DataFrame:
    """Conserva el run más reciente por (experiment_id, ticker, timeframe).

    `session_id` tiene formato YYYYMMDD_HHMMSS → el orden lexicográfico es
    cronológico. Reporta cuántos duplicados se eliminaron vía atributo .attrs.
    """
    if time_col not in runs.columns and '_session_id' in runs.columns:
        time_col = '_session_id'
    key_cols = [c for c in key_cols if c in runs.columns]
    before = len(runs)
    out = (runs.sort_values(time_col)
               .drop_duplicates(subset=key_cols, keep="last")
               .reset_index(drop=True))
    out.attrs["duplicates_removed"] = before - len(out)
    return out


# ---------------------------------------------------------------------------
# [C1]+[C7] Contexto de validación precomputado (una vez por grid search)
# ---------------------------------------------------------------------------

_REGIME_CODES = {"bull": 0, "bear": 1, "flat": 2}


@dataclass(frozen=True)
class ValidationContext:
    """Todo lo que depende SOLO del dataset, precomputado una vez por grid.

    Solo ndarrays y escalares → picklable y barato de enviar a los workers
    de joblib. De solo lectura: compartido por todos los runs.
    """
    ret: np.ndarray            # float64 (n,) — Close.pct_change, [0]=0.0
    val_start: int             # índice del corte cronológico train|val
    val_w_bounds: np.ndarray   # int64 (4,) — offsets de las 3 sub-ventanas en validación
    regime_val: np.ndarray     # int8 (n_val,) — 0=bull, 1=bear, 2=flat
    bh_val: np.ndarray         # float64 (n_val,) — retornos B&H en validación
    roi_bh_val: float
    periods_per_year: int
    years_val: float
    val_frac: float
    signal_delay: int
    cost_rate: float           # commission + slippage (por unidad de turnover)
    min_window_obs: int = 30   # mínimo de obs por sub-ventana para calcular Sharpe


def build_validation_context(df: pd.DataFrame,
                             timeframe: str | None = None,
                             commission: float = 0.001,
                             slippage: float = 0.001,
                             val_frac: float = 0.3,
                             signal_delay: int = 1,
                             periods_per_year: int | None = None,
                             close_col: str = COL_CLOSE) -> ValidationContext:
    """Precomputa el ValidationContext para un dataset. Llamar UNA vez por
    grid search; los runs individuales usan `compute_validation_metrics`.

    `periods_per_year` se deriva del `timeframe` si no se pasa explícito.
    Las ventanas del clasificador de régimen (200d máximo / 60d rango) se
    escalan a barras del timeframe.
    """
    if periods_per_year is None:
        if timeframe is None:
            raise ValueError("Indica timeframe o periods_per_year")
        periods_per_year = periods_per_year_from_timeframe(timeframe)

    close = df[close_col].astype(float)
    ret = close.pct_change().fillna(0.0).to_numpy(dtype=np.float64)

    n = len(df.index)
    val_start = int(np.floor(n * (1.0 - val_frac)))
    n_val = n - val_start

    bars_per_day = max(1, periods_per_year // ANNUAL)
    regime = classify_regime(close, lookback=60 * bars_per_day,
                             max_window=200 * bars_per_day)
    regime_val = regime.map(_REGIME_CODES).to_numpy(dtype=np.int8)[val_start:]

    bh_val = ret[val_start:].copy()
    roi_bh_val = float(np.prod(1.0 + bh_val) - 1.0)

    return ValidationContext(
        ret=ret,
        val_start=val_start,
        val_w_bounds=_subwindow_bounds(n_val, 3),
        regime_val=regime_val,
        bh_val=bh_val,
        roi_bh_val=roi_bh_val,
        periods_per_year=int(periods_per_year),
        years_val=n_val / periods_per_year if periods_per_year else np.nan,
        val_frac=float(val_frac),
        signal_delay=int(signal_delay),
        cost_rate=float(commission) + float(slippage),
    )


def _windowed_sharpes(r: np.ndarray, bounds: np.ndarray,
                      periods_per_year: int, min_obs: int) -> list[float]:
    """Sharpe por sub-ventana usando offsets precomputados."""
    out = []
    for i in range(len(bounds) - 1):
        chunk = r[bounds[i]:bounds[i + 1]]
        out.append(_sharpe_np(chunk, periods_per_year)
                   if chunk.shape[0] >= min_obs else np.nan)
    return out


def compute_validation_metrics(pos: np.ndarray, ctx: ValidationContext) -> dict:
    """
    Métricas de validación de UN run (hot path, 100% numpy).

    Parámetros:
      pos  posición objetivo por barra (float, {-1, 0, 1}) ANTES del delay —
           salida de `_target_position(signal, position_type)`
      ctx  ValidationContext precomputado del dataset

    Devuelve dict plano de escalares (bools como 1.0/0.0 para tipado Parquet
    estable) listo para fusionar en el dict de resultado del run.
    """
    n = ctx.ret.shape[0]
    d = ctx.signal_delay

    pos = np.asarray(pos, dtype=np.float64)
    pos = np.nan_to_num(pos, nan=0.0)
    pos_exec = np.zeros(n, dtype=np.float64)
    if d > 0:
        pos_exec[d:] = pos[:n - d]
    else:
        pos_exec[:] = pos[:n]

    turnover = np.abs(np.diff(pos_exec, prepend=0.0))
    # strat_ret in-place sobre pos_exec (ya no se necesita tras el turnover)
    strat_ret = pos_exec
    np.multiply(strat_ret, ctx.ret, out=strat_ret)
    if ctx.cost_rate != 0.0:
        strat_ret -= turnover * ctx.cost_rate

    # El corte train|val es contiguo → slices (vistas), no boolean indexing
    r_train = strat_ret[:ctx.val_start]
    r_val = strat_ret[ctx.val_start:]

    # Sharpe train/val (+ Sharpe de validación POR PERÍODO, insumo del DSR)
    sharpe_train = _sharpe_np(r_train, ctx.periods_per_year)
    sharpe_val = _sharpe_np(r_val, ctx.periods_per_year)
    sr_val_period = np.nan
    if r_val.shape[0] >= 2:
        nv = r_val.shape[0]
        mv = r_val.sum() / nv
        varv = (float(np.dot(r_val, r_val)) - nv * mv * mv) / (nv - 1)
        if varv > 0:
            sr_val_period = float(mv / np.sqrt(varv))

    # Sub-ventanas de validación [C1]
    w = _windowed_sharpes(r_val, ctx.val_w_bounds, ctx.periods_per_year,
                          ctx.min_window_obs)

    # Equity de validación → ROI y MaxDD
    if r_val.shape[0]:
        eq = np.cumprod(1.0 + r_val)
        roi_val = float(eq[-1] - 1.0)
        maxdd_val = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    else:
        roi_val, maxdd_val = np.nan, np.nan

    # Régimen externo [C6]
    reg_sharpes = {}
    for name, code in _REGIME_CODES.items():
        reg_sharpes[f"sharpe_{name}"] = _sharpe_np(
            r_val[ctx.regime_val == code], ctx.periods_per_year)
    bb = [reg_sharpes["sharpe_bull"], reg_sharpes["sharpe_bear"]]
    bb = [v for v in bb if not np.isnan(v)]
    min_regime_sharpe = min(bb) if bb else np.nan

    # IR vs Buy&Hold por sub-ventana [C4]
    active = r_val - ctx.bh_val
    irs = _windowed_sharpes(active, ctx.val_w_bounds, ctx.periods_per_year,
                            ctx.min_window_obs)
    windows_beating_bh = sum(1 for x in irs if not np.isnan(x) and x > 0)

    # Frecuencia (entrada+salida = 1 trade); el objetivo 15-60 ops/año se
    # evalúa a nivel portfolio [S2], aquí solo el mínimo estadístico
    n_trades_total = int(round(turnover[ctx.val_start:].sum() / 2.0))
    n_trades_year = n_trades_total / ctx.years_val if ctx.years_val else np.nan

    return {
        "sharpe_train": sharpe_train,
        "sharpe_val": sharpe_val,
        "sr_val_period": sr_val_period,
        "sharpe_val_w1": w[0], "sharpe_val_w2": w[1], "sharpe_val_w3": w[2],
        "roi_val": roi_val,
        "roi_bh_val": ctx.roi_bh_val,
        "maxdd_val": maxdd_val,
        "n_trades_total": n_trades_total,
        "n_trades_year": n_trades_year,
        "passes_min_trades": 1.0 if min_trades_statistical(n_trades_total) else 0.0,
        **reg_sharpes,
        "min_regime_sharpe": min_regime_sharpe,
        "windows_beating_bh": windows_beating_bh,
        "passes_ir_2of3": 1.0 if windows_beating_bh >= 2 else 0.0,
    }


def backtest_with_validation(df: pd.DataFrame,
                             position: pd.Series,
                             close_col: str = COL_CLOSE,
                             commission_rate: float = 0.001,
                             slippage_rate: float = 0.001,
                             val_frac: float = 0.3,
                             signal_delay: int = 1,
                             periods_per_year: int = ANNUAL) -> dict:
    """
    Wrapper standalone (notebooks / re-ejecuciones puntuales): construye un
    ValidationContext ad-hoc y delega en `compute_validation_metrics`.

    Parámetros:
      df              OHLCV indexado por fecha (columna `close_col`)
      position        serie de posición objetivo ({-1,0,1}) ANTES del delay
      commission_rate coste por unidad de turnover [C7: separado de slippage]
      slippage_rate   coste adicional aplicado solo en cambios de posición
      signal_delay    1 = estándar; pasar 2 para el test de lookahead (B5.3)

    En el grid search NO uses este wrapper: usa build_validation_context una
    vez + compute_validation_metrics por run.
    """
    ctx = build_validation_context(
        df, commission=commission_rate, slippage=slippage_rate,
        val_frac=val_frac, signal_delay=signal_delay,
        periods_per_year=periods_per_year, close_col=close_col,
    )
    pos = position.reindex(df.index).fillna(0).to_numpy(dtype=np.float64)
    return compute_validation_metrics(pos, ctx)


# ---------------------------------------------------------------------------
# [S1 aplicado] Cálculo de estabilidad para una tabla de resultados
# ---------------------------------------------------------------------------

def compute_stability_table(results: pd.DataFrame,
                            group_col: str = "ticker",
                            strategy_col: str = "strategy_name",
                            sharpe_floor: float = 0.5) -> pd.DataFrame:
    """
    Añade rank_train, rank_val, n_universe y stability (asimétrica) a un
    DataFrame con columnas sharpe_train y sharpe_val (export Parquet).

    IMPORTANTE (universos desiguales): el ranking se calcula por-grupo
    (por activo), sobre el subconjunto de estrategias que EXISTEN para ese
    activo. n_universe se reporta junto al rank — nunca compares ranks de
    universos de tamaño distinto sin mirar n_universe.
    """
    df = results.copy()
    df["rank_train"] = df.groupby(group_col)["sharpe_train"] \
                         .rank(ascending=False, method="min")
    df["rank_val"] = df.groupby(group_col)["sharpe_val"] \
                       .rank(ascending=False, method="min")
    df["n_universe"] = df.groupby(group_col)[strategy_col].transform("count")
    df["stability"] = df.apply(
        lambda r: asymmetric_stability(int(r["rank_train"]), int(r["rank_val"]),
                                       int(r["n_universe"]), r["sharpe_val"],
                                       sharpe_floor)
        if not (np.isnan(r["rank_train"]) or np.isnan(r["rank_val"])) else 0.0,
        axis=1)
    return df


# ---------------------------------------------------------------------------
# Regla de decisión pre-registrada (única, para el Bloque 8)
# ---------------------------------------------------------------------------

def decision_rule(row: pd.Series,
                  frac_assets_positive: float,
                  survives_double_commission: bool) -> str:
    """
    Regla ÚNICA pre-registrada. No se ajusta después de ver los datos.

    SELECCIONAR si TODO se cumple:
      sharpe_val > 0.5
      stability >= 0.8            (asimétrica)
      frac_assets_positive >= 0.7 (cross-asset)
      min_regime_sharpe > 0       (no colapsa en bear)
      passes_ir_2of3              (bate B&H en ≥2/3 sub-ventanas)
      survives_double_commission  (ROI>0 con comisión 0,2%)
      passes_min_trades           (≥15 trades totales)

    ZONA_GRIS si: sharpe_val > 0.5 y stability >= 0.6 y falla ≤ 2 criterios.
    DESCARTAR en el resto.
    """
    checks = {
        "sharpe": row.get("sharpe_val", np.nan) > 0.5,
        "stability": row.get("stability", 0) >= 0.8,
        "cross_asset": frac_assets_positive >= 0.7,
        "regime": row.get("min_regime_sharpe", np.nan) > 0,
        "ir": bool(row.get("passes_ir_2of3", False)),
        "commission": bool(survives_double_commission),
        "min_trades": bool(row.get("passes_min_trades", False)),
    }
    if all(checks.values()):
        return "SELECCIONAR"
    fails = sum(1 for v in checks.values() if not v)
    if checks["sharpe"] and row.get("stability", 0) >= 0.6 and fails <= 2:
        return "ZONA_GRIS"
    return "DESCARTAR"
