"""
Módulo de grid search
Optimización de hiperparámetros para estrategias de trading
"""

import time
import gc
import datetime
import pandas as pd
from itertools import product
from .backtesting import backtest_strategy

# MLflow (importación condicional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow no disponible - desactivando tracking")


def strategy_grid_search(df, strategy_configs, use_mlflow=True, ticker='BTCUSDT', timeframe='1h', 
                         experiment_name='default'):
    """
    Grid Search para optimizar parámetros de estrategias de trading
    Calcula indicadores dinámicamente según configuración
    
    Parameters:
    -----------
    df : DataFrame
        Datos OHLCV básicos (open, high, low, close, volume, returns)
        NO necesita indicadores pre-calculados
    strategy_configs : list
        Lista de configuraciones de estrategia a probar
        Soporta estrategias individuales y combinadas
    use_mlflow : bool
        Si True, registra experimentos en MLflow
    ticker : str
        Símbolo del activo (ej: 'BTCUSDT', 'ETHUSDT', 'SOLUSDT')
        Se normaliza a mayúsculas. Se usa como tag en MLflow
        Default: 'BTCUSDT'
    timeframe : str
        Timeframe de los datos (ej: '1h', '4h', '1d', '15m')
        Se normaliza a minúsculas. Se usa como tag en MLflow
        Default: '1h'
    experiment_name : str
        Nombre del experimento en MLflow
        Default: 'default'
    
    Returns:
    --------
    DataFrame con resultados de todas las combinaciones
    Incluye columnas 'ticker' y 'timeframe' para análisis multi-asset
    """
    all_results = []
    
    # Normalizar ticker y timeframe
    ticker_normalized = ticker.upper()
    timeframe_normalized = timeframe.lower()
    
    print(f"\n🎯 Asset: {ticker_normalized} | Timeframe: {timeframe_normalized}")
    print(f"📊 Experimento MLflow: {experiment_name}")
    
    # Limpiar cache de indicadores al inicio (para nuevo ticker/timeframe)
    from .indicators import clear_indicator_cache
    clear_indicator_cache()
    
    # Configurar MLflow
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            from .utils.paths import get_mlruns_dir
            mlruns_path = get_mlruns_dir()
            
            mlflow.set_tracking_uri(f"file:{mlruns_path}")
            mlflow.set_experiment(experiment_name)
            print(f"✓ MLflow configurado: {mlruns_path}")
        except Exception as e:
            print(f"⚠️  Error configurando MLflow: {e}")
            use_mlflow = False
    elif use_mlflow and not MLFLOW_AVAILABLE:
        print("⚠️  MLflow no disponible - tracking desactivado")
        use_mlflow = False
    
    # Calcular total de experimentos
    total_experiments = 0
    for config in strategy_configs:
        if config.get('type') == 'combo':
            # Estrategia combinada
            indicators = config['indicators']
            n_combos = 1
            for ind in indicators:
                n_combos *= len(list(product(*ind['params_grid'].values())))
            n_combos *= len(config.get('combination_methods', ['AND']))
            n_combos *= len(config.get('position_types', ['long']))
            total_experiments += n_combos
        else:
            # Estrategia individual
            params_grid = config['params_grid']
            param_values = list(params_grid.values())
            total_experiments += len(list(product(*param_values)))
    
    print(f"📊 Total de experimentos a ejecutar: {total_experiments}")
    
    experiment_count = 0
    start_time = time.time()
    
    # Iterar sobre cada configuración de estrategia
    for strategy_config in strategy_configs:
        strategy_name = strategy_config['name']
        
        if strategy_config.get('type') == 'combo':
            # ========== ESTRATEGIA COMBINADA ==========
            print(f"\n📊 Estrategia Combinada: {strategy_name}")
            
            indicators = strategy_config['indicators']
            
            # Generar todas las combinaciones de parámetros para cada indicador
            indicators_param_combos = []
            for ind_config in indicators:
                indicator_name = ind_config['indicator']
                params_grid = ind_config['params_grid']
                
                param_names = list(params_grid.keys())
                param_values = list(params_grid.values())
                param_combos = list(product(*param_values))
                
                combos_list = [dict(zip(param_names, combo)) for combo in param_combos]
                
                indicators_param_combos.append({
                    'indicator': indicator_name,
                    'combos': combos_list
                })
            
            # Generar producto cartesiano de todos los indicadores
            all_indicator_combos = list(product(*[ind['combos'] for ind in indicators_param_combos]))
            
            combination_methods = strategy_config.get('combination_methods', ['AND'])
            position_types = strategy_config.get('position_types', ['long'])
            
            total_combos = len(all_indicator_combos) * len(combination_methods) * len(position_types)
            print(f"   Combinaciones totales: {total_combos}")
            
            for indicator_params_set in all_indicator_combos:
                # Construir configuración de indicadores
                indicators_combo = []
                for idx, ind_params in enumerate(indicator_params_set):
                    indicators_combo.append({
                        'indicator': indicators_param_combos[idx]['indicator'],
                        'params': ind_params
                    })
                
                # Probar con diferentes métodos de combinación
                for combination_method in combination_methods:
                    for position_type in position_types:
                        experiment_count += 1
                        
                        # Ejecutar backtest combinado
                        metrics = backtest_strategy(
                            df=df,
                            indicator=None,
                            params={},
                            position_type=position_type,
                            indicators_combo=indicators_combo,
                            combination_method=combination_method
                        )
                        
                        if metrics is None:
                            continue
                        
                        # Preparar resultado
                        result = {
                            'ticker': ticker_normalized,
                            'timeframe': timeframe_normalized,
                            'strategy_name': strategy_name,
                            'strategy_type': 'combo',
                            'combination_method': combination_method,
                            'position_type': position_type,
                            'n_indicators': len(indicators_combo),
                            **metrics
                        }
                        
                        # Agregar parámetros de cada indicador
                        for idx, ind_combo in enumerate(indicators_combo):
                            indicator_name = ind_combo['indicator']
                            result[f'ind{idx+1}_name'] = indicator_name
                            for param_name, param_value in ind_combo['params'].items():
                                result[f'ind{idx+1}_{param_name}'] = param_value
                        
                        all_results.append(result)
                        
                # Log en MLflow
                if use_mlflow and MLFLOW_AVAILABLE:
                    with mlflow.start_run(run_name=f"{strategy_name}_{experiment_count}"):
                        mlflow.log_param("strategy_name", strategy_name)
                        mlflow.log_param("strategy_type", "combo")
                        mlflow.log_param("combination_method", combination_method)
                        mlflow.log_param("position_type", position_type)
                        mlflow.log_param("n_indicators", len(indicators_combo))
                        
                        for idx, ind_combo in enumerate(indicators_combo):
                            mlflow.log_param(f"ind{idx+1}_name", ind_combo['indicator'])
                            for param_name, param_value in ind_combo['params'].items():
                                mlflow.log_param(f"ind{idx+1}_{param_name}", param_value)
                        
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        
                        mlflow.set_tag("ticker", ticker_normalized)
                        mlflow.set_tag("timeframe", timeframe_normalized)
                        mlflow.set_tag("strategy_type", "combo")
                        # Tag de sesión para agrupar ejecuciones (usa timestamp)
                        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        mlflow.set_tag("session_id", session_id)                        # Progreso
                        if experiment_count % 10 == 0:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / experiment_count
                            remaining = (total_experiments - experiment_count) * avg_time
                            print(f"   [{experiment_count}/{total_experiments}] "
                                  f"Method: {combination_method} | "
                                  f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                                  f"ETA: {remaining:.0f}s")
        
        else:
            # ========== ESTRATEGIA INDIVIDUAL ==========
            indicator = strategy_config['indicator']
            params_grid = strategy_config['params_grid']
            
            # Generar todas las combinaciones de parámetros
            param_names = list(params_grid.keys())
            param_values = list(params_grid.values())
            param_combinations = list(product(*param_values))
            
            print(f"\n📊 Estrategia Individual: {strategy_name}")
            print(f"   Indicador: {indicator}")
            print(f"   Combinaciones: {len(param_combinations)}")
            
            for param_combo in param_combinations:
                experiment_count += 1
                params = dict(zip(param_names, param_combo))
                
                # Extraer position_type si existe
                position_type = params.pop('position_type', 'long')
                
                # Ejecutar backtest
                metrics = backtest_strategy(df, indicator, params, position_type)
                
                if metrics is None:
                    continue
                
                # Preparar resultado
                result = {
                    'ticker': ticker_normalized,
                    'timeframe': timeframe_normalized,
                    'strategy_name': strategy_name,
                    'strategy_type': 'single',
                    'indicator': indicator,
                    'position_type': position_type,
                    **params,
                    **metrics
                }
                
                all_results.append(result)
                
                # Log en MLflow
                if use_mlflow and MLFLOW_AVAILABLE:
                    with mlflow.start_run(run_name=f"{strategy_name}_{experiment_count}"):
                        mlflow.log_param("strategy_name", strategy_name)
                        mlflow.log_param("strategy_type", "single")
                        mlflow.log_param("indicator", indicator)
                        mlflow.log_param("position_type", position_type)
                        for param_name, param_value in params.items():
                            mlflow.log_param(param_name, param_value)
                        
                        for metric_name, metric_value in metrics.items():
                            mlflow.log_metric(metric_name, metric_value)
                        
                        mlflow.set_tag("ticker", ticker_normalized)
                        mlflow.set_tag("timeframe", timeframe_normalized)
                        mlflow.set_tag("strategy_type", "single")
                        # Tag de sesión para agrupar ejecuciones (usa timestamp)
                        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        mlflow.set_tag("session_id", session_id)
                
                # Progreso
                if experiment_count % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / experiment_count
                    remaining = (total_experiments - experiment_count) * avg_time
                    print(f"   [{experiment_count}/{total_experiments}] "
                          f"Sharpe: {metrics['sharpe_ratio']:.2f} | "
                          f"Win%: {metrics['win_rate']:.1%} | "
                          f"ETA: {remaining:.0f}s")
                
                # Liberar memoria cada 100 experimentos
                if experiment_count % 100 == 0:
                    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"\n✓ Grid Search completado en {elapsed:.1f}s")
    print(f"  Experimentos exitosos: {len(all_results)}/{total_experiments}")
    
    # Liberar memoria después del grid search
    results_df = pd.DataFrame(all_results)
    del all_results  # Liberar lista grande
    gc.collect()     # Forzar recolección de basura
    
    if not results_df.empty:
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # Mostrar top 5
        print(f"\n🏆 TOP 5 ESTRATEGIAS (por Sharpe Ratio):")
        print("=" * 80)
        for idx, row in results_df.head(5).iterrows():
            strategy_type = row.get('strategy_type', 'single')
            
            if strategy_type == 'combo':
                indicators_str = ', '.join([row.get(f'ind{i}_name', '?') for i in range(1, row.get('n_indicators', 0) + 1)])
                print(f"\n{idx+1}. {row['strategy_name']} (COMBO: {indicators_str})")
                print(f"   Method: {row['combination_method']}, Position: {row['position_type']}")
            else:
                print(f"\n{idx+1}. {row['strategy_name']} ({row.get('indicator', 'N/A')})")
                print(f"   Position: {row['position_type']}, Period: {row.get('period', 'N/A')}")
            
            print(f"   Sharpe: {row['sharpe_ratio']:.3f} | Win%: {row['win_rate']:.1%} | "
                  f"PF: {row['profit_factor']:.2f} | Return: {row['total_return']:.2%}")
    
    return results_df
