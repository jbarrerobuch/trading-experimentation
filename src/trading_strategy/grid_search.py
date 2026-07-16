"""
Módulo de grid search
Optimización de hiperparámetros para estrategias de trading.

El logging por-experimento de MLflow se ha sustituido por un output estructurado
en columnas (Parquet + manifest JSON, ver utils.results_io), mucho más rápido y
pensado para ser consumido por una web-app de análisis.
"""

import time
import gc
import subprocess
from functools import lru_cache
from itertools import product

import pandas as pd

from .backtesting import backtest_strategy, new_value_cache
from .validation_metrics import build_validation_context
from .utils import results_io


@lru_cache(maxsize=1)
def get_git_commit():
    """Hash del commit actual de git (cacheado: se calcula una sola vez)."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return "unknown"


# ============================================================================
# ENUMERACIÓN DE EXPERIMENTOS
# ============================================================================

def _enumerate_tasks(strategy_configs):
    """
    Genera una lista de "tareas" (un backtest totalmente especificado cada una)
    a partir de las configuraciones de estrategia.
    """
    tasks = []
    for strategy_config in strategy_configs:
        strategy_name = strategy_config['name']

        if strategy_config.get('type') == 'combo':
            indicators = strategy_config['indicators']

            indicators_param_combos = []
            for ind_config in indicators:
                params_grid = ind_config['params_grid']
                param_names = list(params_grid.keys())
                param_values = list(params_grid.values())
                combos_list = [dict(zip(param_names, combo)) for combo in product(*param_values)]
                indicators_param_combos.append({
                    'indicator': ind_config['indicator'],
                    'combos': combos_list
                })

            all_indicator_combos = product(*[ind['combos'] for ind in indicators_param_combos])
            combination_methods = strategy_config.get('combination_methods', ['AND'])
            position_types = strategy_config.get('position_types', ['long'])

            for indicator_params_set in all_indicator_combos:
                indicators_combo = [
                    {'indicator': indicators_param_combos[idx]['indicator'], 'params': ind_params}
                    for idx, ind_params in enumerate(indicator_params_set)
                ]
                for combination_method in combination_methods:
                    for position_type in position_types:
                        tasks.append({
                            'type': 'combo',
                            'strategy_name': strategy_name,
                            'indicators_combo': indicators_combo,
                            'combination_method': combination_method,
                            'position_type': position_type,
                        })
        else:
            indicator = strategy_config['indicator']
            params_grid = strategy_config['params_grid']
            param_names = list(params_grid.keys())
            param_values = list(params_grid.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo))
                position_type = params.pop('position_type', 'long')
                tasks.append({
                    'type': 'single',
                    'strategy_name': strategy_name,
                    'indicator': indicator,
                    'params': params,
                    'position_type': position_type,
                })
    return tasks


def _run_task(df, task, commission, slippage, use_next_open, ticker, timeframe, value_cache,
              validation_ctx=None):
    """Ejecuta un experimento y devuelve el dict de resultado (o None)."""
    if task['type'] == 'combo':
        indicators_combo = task['indicators_combo']
        combination_method = task['combination_method']
        position_type = task['position_type']

        metrics = backtest_strategy(
            df=df, indicator=None, params={}, position_type=position_type,
            indicators_combo=indicators_combo, combination_method=combination_method,
            commission=commission, slippage=slippage, use_next_open=use_next_open,
            value_cache=value_cache, validation_context=validation_ctx,
        )
        if metrics is None:
            return None

        result = {
            'ticker': ticker, 'timeframe': timeframe,
            'strategy_name': task['strategy_name'], 'strategy_type': 'combo',
            'combination_method': combination_method, 'position_type': position_type,
            'n_indicators': len(indicators_combo),
            **metrics,
        }
        for idx, ind_combo in enumerate(indicators_combo):
            result[f'ind{idx+1}_name'] = ind_combo['indicator']
            for param_name, param_value in ind_combo['params'].items():
                result[f'ind{idx+1}_{param_name}'] = param_value
        result['experiment_id'] = results_io.compute_experiment_id(
            'combo', None, {}, position_type,
            indicators_combo=indicators_combo, combination_method=combination_method,
        )
        return result

    # single
    indicator = task['indicator']
    params = task['params']
    position_type = task['position_type']

    metrics = backtest_strategy(
        df, indicator, params, position_type,
        commission=commission, slippage=slippage, use_next_open=use_next_open,
        value_cache=value_cache, validation_context=validation_ctx,
    )
    if metrics is None:
        return None

    result = {
        'ticker': ticker, 'timeframe': timeframe,
        'strategy_name': task['strategy_name'], 'strategy_type': 'single',
        'indicator': indicator, 'position_type': position_type,
        **params, **metrics,
    }
    result['experiment_id'] = results_io.compute_experiment_id(
        'single', indicator, params, position_type,
    )
    return result


def _run_chunk(df, tasks, commission, slippage, use_next_open, ticker, timeframe,
               validation_ctx=None):
    """Ejecuta un lote de tareas con su propio value cache (para paralelo)."""
    value_cache = new_value_cache()
    out = []
    for task in tasks:
        res = _run_task(df, task, commission, slippage, use_next_open, ticker, timeframe,
                        value_cache, validation_ctx)
        if res is not None:
            out.append(res)
    return out


# ============================================================================
# API PÚBLICA
# ============================================================================

def strategy_grid_search(df, strategy_configs, use_mlflow=False, ticker='BTCUSDT', timeframe='1h',
                         experiment_name='default', commission=0.001, slippage=0.0001, use_next_open=True,
                         session_id=None, log_artifacts=False, verbose=True,
                         n_jobs=1, output_dir=None,
                         validation=True, val_frac=0.3, signal_delay=1):
    """
    Grid Search para optimizar parámetros de estrategias de trading.

    Calcula indicadores dinámicamente según configuración. La señal de cada
    indicador se reutiliza entre los distintos ``position_type`` del mismo combo
    de parámetros (cache de señales), evitando recálculos costosos.

    Parameters
    ----------
    df : DataFrame
        Datos OHLCV (open, high, low, close, volume, returns).
    strategy_configs : list
        Configuraciones de estrategia (individuales y/o combinadas).
    ticker, timeframe, experiment_name : str
        Metadatos del dataset/experimento.
    commission, slippage : float
        Aceptados por compatibilidad (no aplicados al PnL en esta versión).
    use_next_open : bool
        Si True, ejecuta al Open de la siguiente vela (más realista).
    session_id : str, optional
        ID de sesión; si None se genera uno por timestamp.
    n_jobs : int
        1 = secuencial (default). >1 o -1 = paralelo con joblib (procesos),
        repartiendo los experimentos entre núcleos.
    output_dir : str or Path, optional
        Si se indica, escribe la tabla de resultados (Parquet) y el manifest
        (JSON) vía utils.results_io.
    validation : bool
        Si True (default), añade a cada run las métricas de validación
        anti-overfitting (sharpe_train/val, sub-ventanas, régimen, IR vs B&H)
        vía validation_metrics. El contexto se precomputa una sola vez.
    val_frac : float
        Fracción final del dataset reservada como validación (corte cronológico).
    signal_delay : int
        Delay de ejecución de las métricas de validación. 1 = estándar sin
        lookahead; 2 = test de lookahead (re-ejecución del Bloque 5.3).
    use_mlflow, log_artifacts : DEPRECATED
        Ignorados. El logging se hace ahora vía output estructurado (output_dir).

    Returns
    -------
    DataFrame con resultados de todas las combinaciones (ordenado por sharpe_ratio).
    """
    if use_mlflow or log_artifacts:
        if verbose:
            print("ℹ️  MLflow/log_artifacts están deprecados en grid_search; "
                  "usa output_dir para el output estructurado (Parquet).")

    ticker_normalized = ticker.upper()
    timeframe_normalized = timeframe.lower()

    if session_id is None:
        session_id = results_io.make_session_id()

    # Enumerar experimentos
    tasks = _enumerate_tasks(strategy_configs)
    total_experiments = len(tasks)

    # Contexto de validación: todo lo que depende solo del dataset se
    # precomputa UNA vez (retornos, régimen, máscaras train/val, B&H).
    validation_ctx = None
    if validation:
        validation_ctx = build_validation_context(
            df, timeframe=timeframe_normalized,
            commission=commission, slippage=slippage,
            val_frac=val_frac, signal_delay=signal_delay,
        )

    if verbose:
        print(f"\n🎯 Asset: {ticker_normalized} | Timeframe: {timeframe_normalized}")
        print(f"📊 Usando {len(df)} velas para optimización")
        print(f"📊 Total de experimentos a ejecutar: {total_experiments}")

    start_time = time.time()
    all_results = []

    # El paralelismo solo compensa con grids grandes (el spawn de procesos en
    # Windows y el pickling del DataFrame tienen coste fijo). Para grids pequeños
    # se ejecuta secuencialmente aunque se pida n_jobs!=1.
    _PARALLEL_MIN_EXPERIMENTS = 500

    if n_jobs and n_jobs != 1 and total_experiments >= _PARALLEL_MIN_EXPERIMENTS:
        # ---- Ejecución paralela (joblib, procesos) ----
        try:
            from joblib import Parallel, delayed, cpu_count
        except ImportError:
            if verbose:
                print("⚠️  joblib no disponible; ejecutando secuencialmente.")
            Parallel = None

        if Parallel is not None:
            import math
            n_workers = cpu_count() if n_jobs in (-1, 0) else n_jobs
            n_workers = max(1, min(n_workers, total_experiments))
            # Chunks CONTIGUOS: cada worker procesa un bloque de tareas seguidas,
            # preservando la localidad del cache de señales (mismos params juntos).
            chunk_size = math.ceil(total_experiments / n_workers)
            chunks = [tasks[i:i + chunk_size] for i in range(0, total_experiments, chunk_size)]
            if verbose:
                print(f"⚡ Paralelo: {len(chunks)} workers")
            results_lists = Parallel(n_jobs=n_workers, prefer="processes")(
                delayed(_run_chunk)(df, chunk, commission, slippage, use_next_open,
                                    ticker_normalized, timeframe_normalized, validation_ctx)
                for chunk in chunks
            )
            for rl in results_lists:
                all_results.extend(rl)
        else:
            n_jobs = 1
    elif n_jobs and n_jobs != 1:
        # Grid pequeño: ignorar paralelismo
        n_jobs = 1

    if n_jobs == 1:
        # ---- Ejecución secuencial (un solo value cache) ----
        value_cache = new_value_cache()
        for i, task in enumerate(tasks, 1):
            res = _run_task(df, task, commission, slippage, use_next_open,
                            ticker_normalized, timeframe_normalized, value_cache,
                            validation_ctx)
            if res is not None:
                all_results.append(res)

            if verbose and i % 50 == 0:
                elapsed = time.time() - start_time
                avg = elapsed / i
                eta = (total_experiments - i) * avg
                print(f"   [{i}/{total_experiments}] ETA: {eta:.0f}s")
            if i % 500 == 0:
                gc.collect()

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n✓ Grid Search completado en {elapsed:.1f}s")
        print(f"  Experimentos exitosos: {len(all_results)}/{total_experiments}")

    results_df = pd.DataFrame(all_results)
    del all_results
    gc.collect()

    if not results_df.empty:
        results_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)

        if verbose:
            print(f"\n🏆 TOP 5 ESTRATEGIAS (por Sharpe Ratio):")
            print("=" * 80)
            for rank, (_, row) in enumerate(results_df.head(5).iterrows()):
                strategy_type = row.get('strategy_type', 'single')
                if strategy_type == 'combo':
                    inds = ', '.join([str(row.get(f'ind{i}_name', '?'))
                                      for i in range(1, int(row.get('n_indicators', 0)) + 1)])
                    print(f"\n{rank+1}. {row['strategy_name']} (COMBO: {inds})")
                    print(f"   Method: {row['combination_method']}, Position: {row['position_type']}")
                else:
                    print(f"\n{rank+1}. {row['strategy_name']} ({row.get('indicator', 'N/A')})")
                    print(f"   Position: {row['position_type']}, Period: {row.get('period', 'N/A')}")
                print(f"   Sharpe: {row['sharpe_ratio']:.3f} | Win%: {row['win_rate']:.1%} | "
                      f"PF: {row['profit_factor']:.2f} | Return: {row['total_return']:.2%}")

    # Output estructurado (reemplazo de MLflow)
    if output_dir is not None and not results_df.empty:
        manifest = results_io.build_manifest(
            session_id=session_id, experiment_name=experiment_name,
            ticker=ticker_normalized, timeframe=timeframe_normalized, df=df,
            commission=commission, slippage=slippage, use_next_open=use_next_open,
            git_commit=get_git_commit(), n_experiments=len(results_df), elapsed_s=elapsed,
            extra={
                'validation': bool(validation),
                'val_frac': float(val_frac),
                'signal_delay': int(signal_delay),
                'periods_per_year': validation_ctx.periods_per_year if validation_ctx else None,
                'n_experiments_enumerated': int(total_experiments),
            },
        )
        paths = results_io.write_results(
            results_df, output_dir, experiment_name=experiment_name,
            ticker=ticker_normalized, timeframe=timeframe_normalized,
            session_id=session_id, manifest=manifest,
        )
        if verbose:
            print(f"\n💾 Resultados: {paths['results']}")
            print(f"💾 Manifest:   {paths.get('manifest')}")

    return results_df
