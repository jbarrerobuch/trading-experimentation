"""
Sistema de Grid Search por Batches
Optimiza memoria y permite checkpoints para experimentos grandes
"""

import time
import datetime
import pandas as pd
from itertools import product, islice
from .grid_search import strategy_grid_search


def batched(iterable, n):
    """
    Divide un iterable en batches de tamaño n
    
    Ejemplo:
        list(batched([1,2,3,4,5], 2)) → [[1,2], [3,4], [5]]
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, n))
        if not batch:
            break
        yield batch


def create_batch_configs(strategy_config, batch_size):
    """
    Divide una configuración de estrategia en múltiples configs más pequeños
    
    Parameters:
    -----------
    strategy_config : dict
        Configuración original con params_grid completo
    batch_size : int
        Número máximo de combinaciones por batch
        
    Returns:
    --------
    list[dict] : Lista de configs batched
    """
    if strategy_config.get('type') == 'combo':
        # Para combos, dividir por combinaciones de indicadores
        indicators_list = strategy_config['indicators']
        combination_methods = strategy_config.get('combination_methods', ['AND'])
        
        # Generar todas las combinaciones de indicadores + métodos
        all_indicator_combos = []
        for combo in product(*[ind['params_grid'].values() for ind in indicators_list]):
            for method in combination_methods:
                all_indicator_combos.append((combo, method))
        
        # Dividir en batches
        batches = list(batched(all_indicator_combos, batch_size))
        
        batch_configs = []
        for batch_idx, batch in enumerate(batches):
            # Crear config reducido para este batch
            # (implementación simplificada - requiere lógica más compleja)
            batch_configs.append({
                **strategy_config,
                '_batch_id': batch_idx,
                '_total_batches': len(batches)
            })
        
        return batch_configs
    
    else:
        # Para estrategias individuales, dividir params_grid
        params_grid = strategy_config['params_grid']
        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        
        # Todas las combinaciones posibles
        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)
        
        if total_combinations <= batch_size:
            # No necesita batching
            return [strategy_config]
        
        # Dividir en batches
        batches = list(batched(all_combinations, batch_size))
        
        batch_configs = []
        for batch_idx, batch_combos in enumerate(batches):
            # Reconstruir params_grid solo con este batch
            batch_params_grid = {}
            for param_idx, param_name in enumerate(param_names):
                unique_values = sorted(set(combo[param_idx] for combo in batch_combos))
                batch_params_grid[param_name] = unique_values
            
            batch_config = {
                **strategy_config,
                'params_grid': batch_params_grid,
                '_batch_id': batch_idx,
                '_total_batches': len(batches),
                '_batch_size': len(batch_combos)
            }
            batch_configs.append(batch_config)
        
        return batch_configs


def batch_grid_search(df, strategy_configs, batch_size=10000, 
                     use_mlflow=True, ticker='BTCUSDT', timeframe='1h',
                     experiment_name='default', save_checkpoints=True):
    """
    Grid Search con división automática en batches
    Optimiza memoria y permite recuperación ante fallos
    
    Parameters:
    -----------
    df : DataFrame
        Datos OHLCV
    strategy_configs : list
        Lista de configuraciones de estrategia
    batch_size : int
        Número máximo de experimentos por batch
        Recomendado: 5K-20K para equilibrar memoria/velocidad
        Default: 10000
    use_mlflow : bool
        Si True, registra en MLflow
    ticker : str
        Símbolo del activo
    timeframe : str
        Timeframe de los datos
    experiment_name : str
        Nombre del experimento en MLflow
    save_checkpoints : bool
        Si True, guarda resultados intermedios en CSV
        Default: True
        
    Returns:
    --------
    DataFrame consolidado con todos los resultados
    
    Example:
    --------
    >>> configs = load_all_strategies()
    >>> # En lugar de 100K runs en una ejecución:
    >>> results = batch_grid_search(df, configs, batch_size=10000)
    >>> # → Ejecuta 10 batches de 10K, cada uno libera memoria
    """
    
    print(f"\n{'='*80}")
    print(f"🔄 BATCH GRID SEARCH")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size:,} experiments per batch")
    print(f"Experiment: {experiment_name}")
    print(f"Ticker: {ticker} | Timeframe: {timeframe}")
    
    # ID único para esta sesión de batches
    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Session ID: {session_id}")
    
    # Dividir cada estrategia en batches
    all_batch_configs = []
    for strategy_config in strategy_configs:
        strategy_name = strategy_config.get('name', 'unnamed')
        batch_configs = create_batch_configs(strategy_config, batch_size)
        
        print(f"\n📊 {strategy_name}: {len(batch_configs)} batch(es)")
        all_batch_configs.extend(batch_configs)
    
    total_batches = len(all_batch_configs)
    print(f"\n🎯 Total batches a ejecutar: {total_batches}")
    
    # Ejecutar cada batch
    all_results = []
    start_time = time.time()
    
    for batch_idx, batch_config in enumerate(all_batch_configs, 1):
        batch_start = time.time()
        strategy_name = batch_config.get('name', 'unnamed')
        batch_id = batch_config.get('_batch_id', 0)
        total_strategy_batches = batch_config.get('_total_batches', 1)
        
        print(f"\n{'─'*80}")
        print(f"📦 BATCH {batch_idx}/{total_batches}")
        print(f"   Strategy: {strategy_name} (batch {batch_id+1}/{total_strategy_batches})")
        print(f"{'─'*80}")
        
        try:
            # Ejecutar grid search para este batch
            # (pasamos configs individuales, no lista)
            batch_results = strategy_grid_search(
                df=df,
                strategy_configs=[batch_config],
                use_mlflow=use_mlflow,
                ticker=ticker,
                timeframe=timeframe,
                experiment_name=experiment_name
            )
            
            # Agregar metadatos de batch
            if not batch_results.empty:
                batch_results['_batch_id'] = batch_idx
                batch_results['_session_id'] = session_id
                all_results.append(batch_results)
                
                batch_elapsed = time.time() - batch_start
                total_elapsed = time.time() - start_time
                avg_time_per_batch = total_elapsed / batch_idx
                eta = avg_time_per_batch * (total_batches - batch_idx)
                
                print(f"\n✓ Batch completado en {batch_elapsed:.1f}s")
                print(f"  Experimentos en batch: {len(batch_results)}")
                print(f"  Total acumulado: {sum(len(r) for r in all_results):,}")
                print(f"  ETA total: {eta:.0f}s ({eta/60:.1f}min)")
                
                # Guardar checkpoint
                if save_checkpoints:
                    checkpoint_file = f"checkpoint_{experiment_name}_{session_id}_batch{batch_idx}.csv"
                    batch_results.to_csv(checkpoint_file, index=False)
                    print(f"  💾 Checkpoint guardado: {checkpoint_file}")
            
            else:
                print(f"\n⚠️  Batch {batch_idx} sin resultados válidos")
        
        except Exception as e:
            print(f"\n❌ ERROR en batch {batch_idx}: {str(e)}")
            print(f"   Continuando con siguiente batch...")
            continue
    
    # Consolidar todos los resultados
    if not all_results:
        print(f"\n⚠️  No se obtuvieron resultados válidos")
        return pd.DataFrame()
    
    final_results = pd.concat(all_results, ignore_index=True)
    final_results = final_results.sort_values('sharpe_ratio', ascending=False)
    
    total_elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"✅ BATCH GRID SEARCH COMPLETADO")
    print(f"{'='*80}")
    print(f"Total experiments: {len(final_results):,}")
    print(f"Total batches: {total_batches}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"Avg time per batch: {total_elapsed/total_batches:.1f}s")
    
    # Guardar resultados finales
    final_file = f"batch_results_{experiment_name}_{session_id}_FINAL.csv"
    final_results.to_csv(final_file, index=False)
    print(f"\n💾 Resultados finales guardados: {final_file}")
    
    # Mostrar top 5 global
    print(f"\n🏆 TOP 5 ESTRATEGIAS GLOBALES (por Sharpe Ratio):")
    print("=" * 80)
    for idx, row in final_results.head(5).iterrows():
        strategy_type = row.get('strategy_type', 'single')
        batch_id = row.get('_batch_id', '?')
        
        if strategy_type == 'combo':
            indicators_str = ', '.join([row.get(f'ind{i}_name', '?') 
                                       for i in range(1, row.get('n_indicators', 0) + 1)])
            print(f"\n{idx+1}. {row['strategy_name']} (COMBO: {indicators_str}) [Batch {batch_id}]")
            print(f"   Method: {row['combination_method']}, Position: {row['position_type']}")
        else:
            print(f"\n{idx+1}. {row['strategy_name']} ({row.get('indicator', 'N/A')}) [Batch {batch_id}]")
            print(f"   Position: {row['position_type']}, Period: {row.get('period', 'N/A')}")
        
        print(f"   Sharpe: {row['sharpe_ratio']:.3f} | Win%: {row['win_rate']:.1%} | "
              f"Return: {row['total_return']:.2%} | MaxDD: {row['max_drawdown']:.2%}")
    
    return final_results


def estimate_batch_requirements(strategy_configs, batch_size=10000):
    """
    Estima recursos necesarios para batch grid search
    
    Parameters:
    -----------
    strategy_configs : list
        Configuraciones de estrategia
    batch_size : int
        Tamaño de batch propuesto
        
    Returns:
    --------
    dict con estimaciones de tiempo, memoria, batches
    """
    from itertools import product
    
    total_experiments = 0
    total_batches = 0
    
    print(f"\n{'='*80}")
    print(f"📊 ESTIMACIÓN DE RECURSOS")
    print(f"{'='*80}")
    print(f"Batch size propuesto: {batch_size:,} experiments")
    
    for config in strategy_configs:
        strategy_name = config.get('name', 'unnamed')
        
        if config.get('type') == 'combo':
            # Estimar combos (simplificado)
            indicators = config['indicators']
            n_indicator_combos = 1
            for ind in indicators:
                n_params = 1
                for param_values in ind['params_grid'].values():
                    n_params *= len(param_values)
                n_indicator_combos *= n_params
            
            methods = len(config.get('combination_methods', ['AND']))
            positions = len(config.get('position_types', ['long']))
            n_experiments = n_indicator_combos * methods * positions
        else:
            # Estrategia individual
            params_grid = config['params_grid']
            param_values = list(params_grid.values())
            param_combinations = list(product(*param_values))
            
            positions = len(config.get('position_types', ['long']))
            n_experiments = len(param_combinations) * positions
        
        n_batches = (n_experiments + batch_size - 1) // batch_size  # ceil division
        
        print(f"\n{strategy_name}:")
        print(f"  Experiments: {n_experiments:,}")
        print(f"  Batches: {n_batches}")
        
        total_experiments += n_experiments
        total_batches += n_batches
    
    # Estimaciones
    avg_time_per_exp = 0.015  # 15ms por experimento (post-optimización)
    total_time_seconds = total_experiments * avg_time_per_exp
    
    memory_per_batch = batch_size * 0.5  # ~0.5KB por resultado
    peak_memory_mb = memory_per_batch / 1024  # Memoria peak por batch
    
    print(f"\n{'─'*80}")
    print(f"📈 TOTALES:")
    print(f"  Total experiments: {total_experiments:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Estimated time: {total_time_seconds/60:.1f} min ({total_time_seconds/3600:.1f} hrs)")
    print(f"  Peak memory per batch: ~{peak_memory_mb:.1f} MB")
    print(f"  Total results size: ~{total_experiments * 0.5 / 1024:.1f} MB")
    
    return {
        'total_experiments': total_experiments,
        'total_batches': total_batches,
        'estimated_time_minutes': total_time_seconds / 60,
        'peak_memory_mb': peak_memory_mb,
        'batch_size': batch_size
    }
