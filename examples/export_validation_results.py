"""
Export v2: consolida los Parquet del grid search en el CSV maestro que consume
la sesión de análisis (docs/PROMPT_ANALISIS_v2.md).

Pipeline (Bloques 0 y 2 del prompt, pre-calculados):
  1. Glob de results_*.parquet en --output-dir → concat, con session_id y
     metadatos (val_frac, signal_delay, commission, n_experiments_enumerated)
     tomados del manifest de cada sesión.
  2. Deduplicación por (experiment_id, ticker, timeframe), keep más reciente.
  3. Ranking stability asimétrica por activo (compute_stability_table).
  4. Significancia: t-stat + Bonferroni por run, y DSR aproximado
     (var_sr_trials = var(sr_val_period) del export completo;
      n_trials = suma de n_experiments_enumerated de los manifests).
  5. frac_assets_positive: fracción de tickers con sharpe_val > 0 para la
     misma configuración (experiment_id) — cross-asset.
  6. survives_double_commission: merge con la re-ejecución a commission=0.002
     (--double-commission-dir); sin ella queda NaN → PENDIENTE (bloquea
     SELECCIONAR, permite ZONA_GRIS).
  7. decision_rule() → columna `decision`.

Uso:
    python examples/export_validation_results.py --output-dir results/
        [--double-commission-dir results_2x/] [--out export_v2.csv]
        [--sharpe-floor 0.5]
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]

# Agregar src/ al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / 'src'))

from trading_strategy.validation_metrics import (  # noqa: E402
    dedupe_runs,
    compute_stability_table,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_tstat_bonferroni,
    decision_rule,
)
from trading_strategy.utils import results_io  # noqa: E402


REQUIRED_COLS = [
    'sharpe_train', 'sharpe_val', 'sr_val_period',
    'sharpe_val_w1', 'sharpe_val_w2', 'sharpe_val_w3',
    'roi_val', 'roi_bh_val', 'maxdd_val',
    'n_trades_total', 'n_trades_year', 'passes_min_trades',
    'sharpe_bull', 'sharpe_bear', 'sharpe_flat', 'min_regime_sharpe',
    'windows_beating_bh', 'passes_ir_2of3',
]


def _session_id_from_filename(path: Path) -> str:
    """results_{experiment}_{ticker}_{timeframe}_{session_id}.parquet
    donde session_id empieza por YYYYMMDD_HHMMSS (posible sufijo _bNNNN)."""
    stem = path.stem  # sin .parquet
    parts = stem.split('_')
    # Buscar el token de 8 dígitos (fecha) y unir desde ahí
    for i, tok in enumerate(parts):
        if len(tok) == 8 and tok.isdigit():
            return '_'.join(parts[i:])
    return stem


def load_sessions(output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Carga todos los Parquet de un directorio + metadatos de sus manifests.

    Devuelve (DataFrame concatenado con columna session_id, dict de manifests
    por session_id).
    """
    parquets = sorted(output_dir.glob('results_*.parquet'))
    if not parquets:
        raise FileNotFoundError(f"No hay results_*.parquet en {output_dir}")

    manifests = {}
    for mf in output_dir.glob('manifest_*.json'):
        m = results_io.read_manifest(mf)
        manifests[m.get('session_id', mf.stem)] = m

    frames = []
    for p in parquets:
        df = results_io.read_results(p)
        session_id = _session_id_from_filename(p)
        df['session_id'] = session_id
        # El manifest se escribe con el session_id pasado al grid search
        man = manifests.get(session_id, {})
        df['val_frac'] = man.get('val_frac', np.nan)
        df['signal_delay'] = man.get('signal_delay', np.nan)
        df['commission'] = man.get('commission', np.nan)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out, manifests


def add_significance(df: pd.DataFrame, n_trials: int, manifests: dict) -> pd.DataFrame:
    """Bloque 2D: DSR aproximado + t-stat con Bonferroni por run.

    El DSR exacto necesita la serie de retornos (skew/kurtosis); desde el
    export agregado se aproxima con skew=0, kurt=3 (normal). El t-stat con
    Bonferroni es más conservador y no necesita la serie.
    """
    df = df.copy()
    sr = df['sr_val_period']
    var_sr_trials = float(sr.var()) if sr.notna().sum() > 2 else np.nan
    sr0 = expected_max_sharpe(n_trials, var_sr_trials) if var_sr_trials > 0 else 0.0
    df['sr0_expected_max'] = sr0

    # T = nº de obs de validación ≈ years_val × periods_per_year (del manifest)
    ppy = np.nan
    years_val = np.nan
    for m in manifests.values():
        if m.get('periods_per_year'):
            ppy = m['periods_per_year']
            rows = m.get('data_rows')
            vf = m.get('val_frac')
            if rows and vf is not None:
                years_val = rows * vf / ppy
            break

    T = int(years_val * ppy) if not (np.isnan(years_val) or np.isnan(ppy)) else 0
    df['dsr_approx'] = [
        probabilistic_sharpe_ratio(s, sr0, T, skew=0.0, kurt=3.0)
        if T >= 3 and not np.isnan(s) else np.nan
        for s in sr
    ]

    bonf = [sharpe_tstat_bonferroni(s, years_val, n_trials)
            for s in df['sharpe_val']]
    df['t_stat'] = [b['t'] for b in bonf]
    df['p_bonferroni'] = [b['p_bonferroni'] for b in bonf]
    df['bonferroni_significant'] = [float(b['significant']) for b in bonf]
    df['n_trials'] = n_trials
    return df


def add_cross_asset(df: pd.DataFrame) -> pd.DataFrame:
    """Fracción de tickers con sharpe_val > 0 para la misma configuración."""
    df = df.copy()
    g = df.groupby('experiment_id')['sharpe_val']
    frac = g.transform(lambda s: float((s > 0).mean()))
    df['frac_assets_positive'] = frac
    df['n_assets_tested'] = g.transform('count')
    return df


def add_double_commission(df: pd.DataFrame, dc_dir: Path | None) -> pd.DataFrame:
    """survives_double_commission: ROI de validación > 0 en la re-ejecución
    con commission=0.002 (Bloque 5.4). Sin re-ejecución → NaN (PENDIENTE)."""
    df = df.copy()
    if dc_dir is None:
        df['survives_double_commission'] = np.nan
        return df
    dc, _ = load_sessions(dc_dir)
    dc = dedupe_runs(dc)
    dc = dc[['experiment_id', 'ticker', 'timeframe', 'roi_val']].rename(
        columns={'roi_val': 'roi_val_2x'})
    df = df.merge(dc, on=['experiment_id', 'ticker', 'timeframe'], how='left')
    df['survives_double_commission'] = np.where(
        df['roi_val_2x'].isna(), np.nan, (df['roi_val_2x'] > 0).astype(float))
    return df


def apply_decision(df: pd.DataFrame) -> pd.DataFrame:
    """Regla única pre-registrada (Bloque 8). Los PENDIENTES (NaN en
    survives_double_commission) cuentan como criterio fallido → bloquean
    SELECCIONAR pero permiten ZONA_GRIS, como exige el prompt."""
    decisions = []
    for _, row in df.iterrows():
        frac = row.get('frac_assets_positive', np.nan)
        frac = 0.0 if np.isnan(frac) else frac
        sdc = row.get('survives_double_commission', np.nan)
        sdc_bool = bool(sdc) if not np.isnan(sdc) else False
        decisions.append(decision_rule(row, frac, sdc_bool))
    df = df.copy()
    df['decision'] = decisions
    df.loc[df['survives_double_commission'].isna(), 'commission_test'] = 'PENDIENTE'
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', required=True, type=Path,
                    help='Directorio con results_*.parquet + manifest_*.json')
    ap.add_argument('--double-commission-dir', type=Path, default=None,
                    help='Directorio con la re-ejecución a commission=0.002')
    ap.add_argument('--out', default='validation_export_v2.csv',
                    help='CSV maestro de salida')
    ap.add_argument('--sharpe-floor', type=float, default=0.5,
                    help='Puerta de calidad de la stability asimétrica')
    args = ap.parse_args()

    print(f"📂 Leyendo sesiones de {args.output_dir} ...")
    df, manifests = load_sessions(args.output_dir)
    print(f"   {len(df)} runs en {df['session_id'].nunique()} sesiones")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"⚠️  Faltan columnas de validación: {missing}")
        print("   Re-ejecuta el grid con validation=True (Carencia 1). "
              "El análisis quedará bloqueado en el Bloque 0.2.")

    # 0.1 Dedupe
    df = dedupe_runs(df)
    print(f"   Dedupe: {df.attrs['duplicates_removed']} duplicados eliminados "
          f"→ {len(df)} runs")

    # n_trials real: lo enumerado por diseño, no las filas exportadas
    n_trials = sum(int(m.get('n_experiments_enumerated', m.get('n_experiments', 0)))
                   for m in manifests.values())
    n_trials = max(n_trials, len(df))
    print(f"   n_trials (grid enumerado): {n_trials}")

    if not missing:
        # 2A stability + 2D significancia + 2B cross-asset + 5.4 comisión doble
        df = compute_stability_table(df, sharpe_floor=args.sharpe_floor)
        df = add_significance(df, n_trials, manifests)
        df = add_cross_asset(df)
        df = add_double_commission(df, args.double_commission_dir)
        df = apply_decision(df)

        counts = df['decision'].value_counts()
        print("\n📋 Decisión pre-registrada:")
        for k in ('SELECCIONAR', 'ZONA_GRIS', 'DESCARTAR'):
            print(f"   {k:12s}: {counts.get(k, 0)}")
        if args.double_commission_dir is None:
            print("   ⚠️  survives_double_commission = PENDIENTE en todos los runs"
                  " (sin --double-commission-dir): nadie puede SELECCIONAR aún.")

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"\n💾 Export maestro: {out_path.resolve()} ({len(df)} filas, "
          f"{len(df.columns)} columnas)")


if __name__ == '__main__':
    main()
