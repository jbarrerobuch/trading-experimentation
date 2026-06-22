"""
Resultados de grid search en formato estructurado (Parquet + manifest JSON).

Reemplaza el logging por-experimento de MLflow por un output columnar y tipado,
pensado para ser consumido por una web-app de análisis (DuckDB / Polars / pandas):

- results_{experiment}_{ticker}_{timeframe}_{session_id}.parquet
    Una fila por experimento (ids + params + métricas), columnas tipadas.
- manifest_{session_id}.json
    Metadatos de nivel sesión calculados UNA sola vez.

Cada fila lleva un ``experiment_id`` estable (hash de la estrategia + params +
position_type) para poder unir trades bajo demanda sin re-ejecutar.
"""

import os
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


# Columnas que NO son parámetros de estrategia (para construir el experiment_id
# solo a partir de los campos identificativos).
_NON_PARAM_COLS = {
    'ticker', 'timeframe', 'strategy_name', 'strategy_type', 'indicator',
    'combination_method', 'position_type', 'n_indicators', 'params',
    'n_transactions',
}


def _canonical(value: Any) -> str:
    """Representación canónica y estable de un valor escalar."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def compute_experiment_id(
    strategy_type: str,
    indicator: Optional[str],
    params: Dict[str, Any],
    position_type: str,
    indicators_combo: Optional[List[Dict[str, Any]]] = None,
    combination_method: Optional[str] = None,
) -> str:
    """
    Genera un identificador estable (12 hex) para un experimento.

    Determinista respecto a la estrategia, los parámetros y el position_type,
    de modo que la misma configuración produce siempre el mismo id.
    """
    parts: List[str] = [strategy_type or 'single', f"pos={position_type}"]

    if indicators_combo is not None:
        parts.append(f"method={combination_method}")
        for ind in indicators_combo:
            ind_params = ind.get('params', {})
            params_str = ','.join(f"{k}={_canonical(v)}" for k, v in sorted(ind_params.items()))
            parts.append(f"{ind['indicator']}({params_str})")
    else:
        parts.append(f"ind={indicator}")
        params_str = ','.join(f"{k}={_canonical(v)}" for k, v in sorted((params or {}).items()))
        parts.append(params_str)

    canonical = '|'.join(parts)
    return hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:12]


def make_session_id() -> str:
    """ID de sesión basado en timestamp (YYYYMMDD_HHMMSS)."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def build_manifest(
    *,
    session_id: str,
    experiment_name: str,
    ticker: str,
    timeframe: str,
    df: Optional[pd.DataFrame] = None,
    commission: float = 0.001,
    slippage: float = 0.0001,
    use_next_open: bool = True,
    git_commit: str = "unknown",
    n_experiments: int = 0,
    elapsed_s: Optional[float] = None,
    framework_version: str = "1.0.0",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye el dict de manifest (metadatos de la sesión)."""
    manifest: Dict[str, Any] = {
        'session_id': session_id,
        'experiment_name': experiment_name,
        'ticker': ticker.upper(),
        'timeframe': timeframe.lower(),
        'commission': commission,
        'slippage': slippage,
        'use_next_open': use_next_open,
        'git_commit': git_commit,
        'n_experiments': int(n_experiments),
        'framework_version': framework_version,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    if elapsed_s is not None:
        manifest['elapsed_s'] = float(elapsed_s)
    if df is not None and not df.empty:
        manifest['data_start_date'] = str(df.index.min())
        manifest['data_end_date'] = str(df.index.max())
        manifest['data_rows'] = int(len(df))
    if extra:
        manifest.update(extra)
    return manifest


def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas con dict/list (p.ej. 'params') a str para Parquet."""
    df = df.copy()
    for col in df.columns:
        sample = df[col].dropna().head(1)
        if len(sample) and isinstance(sample.iloc[0], (dict, list)):
            df[col] = df[col].astype(str)
    return df


def results_filename(experiment_name: str, ticker: str, timeframe: str, session_id: str) -> str:
    return f"results_{experiment_name}_{ticker.upper()}_{timeframe.lower()}_{session_id}.parquet"


def write_results(
    results_df: pd.DataFrame,
    out_dir,
    *,
    experiment_name: str,
    ticker: str,
    timeframe: str,
    session_id: str,
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Escribe la tabla de resultados en Parquet y, si se provee, el manifest JSON.

    Returns
    -------
    dict con las rutas escritas: {'results': ..., 'manifest': ...}
    """
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    results_path = out_dir / results_filename(experiment_name, ticker, timeframe, session_id)
    safe = _sanitize_for_parquet(results_df)
    safe.to_parquet(results_path, index=False)

    paths = {'results': str(results_path)}

    if manifest is not None:
        manifest_path = out_dir / f"manifest_{experiment_name}_{session_id}.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        paths['manifest'] = str(manifest_path)

    return paths


def read_results(path) -> pd.DataFrame:
    """Lee una tabla de resultados Parquet."""
    return pd.read_parquet(path)


def read_manifest(path) -> Dict[str, Any]:
    """Lee un manifest JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
