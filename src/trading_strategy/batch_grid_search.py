"""
Sistema de Grid Search por Batches
Optimiza memoria y permite checkpoints para experimentos grandes
"""

import time
import os
import datetime
import pandas as pd
from pathlib import Path
from itertools import product, islice

# Intentar importar tqdm para barra de progreso
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    
from .grid_search import strategy_grid_search
from .utils.paths import get_project_root


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
                     experiment_name='default', save_checkpoints=True,
                     checkpoint_dir=None, output_file=None):
    """
    Grid Search con división automática en batches
    Optimiza memoria y permite recuperación automática (Auto-Resume)
    
    Parameters:
    -----------
    df : DataFrame
        Datos OHLCV
    strategy_configs : list
        Lista de configuraciones de estrategia
    batch_size : int
        Número máximo de experimentos por batch
    use_mlflow : bool
        Si True, registra en MLflow
    ticker : str
        Símbolo del activo
    timeframe : str
        Timeframe de los datos
    experiment_name : str
        Nombre del experimento. Se usa para agrupar checkpoints.
    save_checkpoints : bool
        Si True, guarda y carga checkpoints automáticamente.
    checkpoint_dir : str or Path, optional
        Directorio de checkpoints.
    output_file : str or Path, optional
        Ruta del archivo final.
        
    Returns:
    --------
    DataFrame consolidado con todos los resultados
    """
    import re
    
    print(f"\n{'='*80}")
    print(f"🔄 BATCH GRID SEARCH (Auto-Resume Enabled)")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size:,} experiments per batch")
    print(f"Experiment: {experiment_name}")
    print(f"Ticker: {ticker} | Timeframe: {timeframe}")
    
    # Generar SIEMPRE un nuevo ID para la ejecución actual
    current_session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Current Run Session ID: {current_session_id}")
    
    # Configurar directorio de checkpoints
    if checkpoint_dir is None:
        checkpoint_dir = get_project_root() / 'checkpoints'
    else:
        checkpoint_dir = Path(checkpoint_dir)
        
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"📂 Directorio de checkpoints: {checkpoint_dir}")
    
    # Dividir cada estrategia en batches
    all_batch_configs = []
    for strategy_config in strategy_configs:
        strategy_name = strategy_config.get('name', 'unnamed')
        batch_configs = create_batch_configs(strategy_config, batch_size)
        
        print(f"\n📊 {strategy_name}: {len(batch_configs)} batch(es)")
        all_batch_configs.extend(batch_configs)
    
    total_batches = len(all_batch_configs)
    print(f"\n🎯 Total batches a ejecutar: {total_batches}")
    
    # ==========================================
    # LÓGICA DE RECUPERACIÓN (AUTO-RESUME)
    # ==========================================
    completed_batches = {} # Map: batch_idx -> (timestamp, filepath, df)
    all_results = []
    
    if save_checkpoints:
        print(f"\n🔍 Buscando checkpoints previos para '{experiment_name}' ({ticker})...")
        
        # Regex para parsear: checkpoint_{experiment_name}_{ticker}_{session_id}_batch{batch_idx}.csv
        # session_id formato: YYYYMMDD_HHMMSS (15 chars)
        # Usamos re.escape para el nombre del experimento y ticker por si tienen caracteres especiales
        filename_pattern = re.compile(rf"checkpoint_{re.escape(experiment_name)}_{re.escape(ticker)}_(\d{{8}}_\d{{6}})_batch(\d+)\.csv")
        
        checkpoint_files = list(checkpoint_dir.glob(f"checkpoint_{experiment_name}_{ticker}_*_batch*.csv")) # pyright: ignore[reportOptionalMemberAccess]
        
        for cp_file in checkpoint_files:
            match = filename_pattern.match(cp_file.name)
            if match:
                cp_session_id = match.group(1)
                batch_idx = int(match.group(2))
                
                # Si encontramos múltiples versiones del mismo batch, nos quedamos con la más reciente (por session_id)
                if batch_idx in completed_batches:
                    if cp_session_id > completed_batches[batch_idx]['session_id']:
                        completed_batches[batch_idx] = {'session_id': cp_session_id, 'file': cp_file}
                else:
                    completed_batches[batch_idx] = {'session_id': cp_session_id, 'file': cp_file}
        
        # Cargar los datos de los checkpoints seleccionados
        if completed_batches:
            print(f"   Encontrados checkpoints para {len(completed_batches)} batches.")
            for batch_idx, info in sorted(completed_batches.items()):
                try:
                    df_cp = pd.read_csv(info['file'])
                    if not df_cp.empty:
                        all_results.append(df_cp)
                        print(f"   ✓ Batch {batch_idx} recuperado de sesión {info['session_id']}")
                    else:
                        # Si está vacío, lo quitamos para que se re-ejecute
                        print(f"   ⚠️ Batch {batch_idx} estaba vacío, se re-ejecutará.")
                        del completed_batches[batch_idx]
                except Exception as e:
                    print(f"   ❌ Error leyendo {info['file'].name}, se re-ejecutará: {e}")
                    del completed_batches[batch_idx]
        else:
            print("   No se encontraron checkpoints previos válidos.")

    # ==========================================
    # EJECUCIÓN
    # ==========================================
    start_time = time.time()
    
    # Identificar batches pendientes
    pending_batches = []
    for idx, config in enumerate(all_batch_configs, 1):
        if idx not in completed_batches:
            pending_batches.append((idx, config))
            
    print(f"\n🚀 Iniciando ejecución de {len(pending_batches)} batches pendientes...")
    
    # Configurar iterador (con tqdm si está disponible)
    iterator = pending_batches
    if TQDM_AVAILABLE:
        iterator = tqdm(pending_batches, desc="Progreso Total", unit="batch", smoothing=0.1)
        
    for batch_idx, batch_config in iterator:
        batch_start = time.time()
        strategy_name = batch_config.get('name', 'unnamed')
        batch_id = batch_config.get('_batch_id', 0)
        total_strategy_batches = batch_config.get('_total_batches', 1)
        
        # Log visual
        info_str = f"Batch {batch_idx}/{total_batches} | Strat: {strategy_name} ({batch_id+1}/{total_strategy_batches})"
        
        if TQDM_AVAILABLE:
            # Actualizar descripción de tqdm para mostrar qué está procesando
            iterator.set_postfix_str(f"curr: {strategy_name}", refresh=True) # pyright: ignore
        else:
            # Fallback a prints si no hay tqdm
            print(f"\n{'─'*80}")
            print(f"📦 {info_str}")
            print(f"{'─'*80}")
        
        try:
            # Ejecutar grid search para este batch
            # Usamos el current_session_id para los nuevos runs
            batch_results = strategy_grid_search(
                df=df,
                strategy_configs=[batch_config],
                use_mlflow=use_mlflow,
                ticker=ticker,
                timeframe=timeframe,
                experiment_name=experiment_name,
                session_id=current_session_id 
            )
            
            # Agregar metadatos de batch
            if not batch_results.empty:
                batch_results['_batch_id'] = batch_idx
                batch_results['_session_id'] = current_session_id
                all_results.append(batch_results)
                
                batch_elapsed = time.time() - batch_start
                # ETA y Timing handled by tqdm automatically if available
                
                if not TQDM_AVAILABLE:
                    # Cálculo manual si no hay tqdm
                    total_elapsed = time.time() - start_time
                    batches_run_so_far = len([b for b in range(1, batch_idx+1) if b not in completed_batches]) # Nota: esto es aproximado si el orden cambia
                    if batches_run_so_far > 0:
                         # Ajustar lógica para pending_batches si fuera necesario, pero para fallback básico está bien
                        pass
                    print(f"\n✓ Batch completado en {batch_elapsed:.1f}s")
                
                # Guardar checkpoint con el ID de la sesión ACTUAL
                if save_checkpoints:
                    filename = f"checkpoint_{experiment_name}_{ticker}_{current_session_id}_batch{batch_idx}.csv"
                    checkpoint_file = checkpoint_dir / filename # pyright: ignore[reportOptionalOperand]
                    batch_results.to_csv(checkpoint_file, index=False)
                    
                    msg = f"  💾 Checkpoint guardado: {filename}"
                    if TQDM_AVAILABLE:
                        tqdm.write(msg) # pyright: ignore
                    else:
                        print(msg)
            
            else:
                msg = f"⚠️  Batch {batch_idx} sin resultados válidos"
                if TQDM_AVAILABLE:
                    tqdm.write(msg) # pyright: ignore
                else:
                    print(f"\n{msg}")
        
        except Exception as e:
            err_msg = f"❌ ERROR en batch {batch_idx}: {str(e)}"
            if TQDM_AVAILABLE:
                tqdm.write(err_msg) # pyright: ignore
                tqdm.write("   Continuando con siguiente batch...") # pyright: ignore
            else:
                print(f"\n{err_msg}")
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
    print(f"  - Recuperados: {len(completed_batches)}")
    print(f"  - Ejecutados: {total_batches - len(completed_batches)}")
    
    # Guardar resultados finales
    if output_file:
        final_file = str(output_file)
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(os.path.abspath(final_file)), exist_ok=True)
    else:
        # Usamos el current_session_id para el archivo final, indicando cuándo se terminó
        final_file = f"batch_results_{experiment_name}_{current_session_id}_FINAL.csv"
        
    final_results.to_csv(final_file, index=False)
    print(f"\n💾 Resultados finales guardados: {final_file}")
    
    # Mostrar top 5 global
    print(f"\n🏆 TOP 5 ESTRATEGIAS GLOBALES (por Sharpe Ratio):")
    print("=" * 80)
    for idx, row in final_results.head(5).iterrows():
        strategy_type = row.get('strategy_type', 'single')
        batch_id = row.get('_batch_id', '?')
        sess_id = row.get('_session_id', '?')
        
        if strategy_type == 'combo':
            indicators_str = ', '.join([row.get(f'ind{i}_name', '?') 
                                       for i in range(1, int(row.get('n_indicators', 0)) + 1)])
            print(f"\n{idx+1}. {row['strategy_name']} (COMBO: {indicators_str}) [Batch {batch_id} | {sess_id}]")
            print(f"   Method: {row['combination_method']}, Position: {row['position_type']}")
        else:
            print(f"\n{idx+1}. {row['strategy_name']} ({row.get('indicator', 'N/A')}) [Batch {batch_id} | {sess_id}]")
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
    # Tiempo base por experimento (backtest loop) - Ajustado a 50ms para ser más conservador
    avg_time_per_exp = 10
    
    # Overhead por batch (setup, logging, I/O, MLflow) - ~2 segundos por batch
    overhead_per_batch = 2.0
    
    total_time_seconds = (total_experiments * avg_time_per_exp) + (total_batches * overhead_per_batch)
    
    memory_per_batch = batch_size * 0.5  # ~0.5KB por resultado
    peak_memory_mb = memory_per_batch / 1024  # Memoria peak por batch
    
    print(f"\n{'─'*80}")
    print(f"📈 TOTALES:")
    print(f"  Total experiments: {total_experiments:,}")
    print(f"  Total batches: {total_batches}")
    print(f"  Estimated time: {total_time_seconds/60:.1f} min ({total_time_seconds/3600:.1f} hrs) per ticker/dataset")
    print(f"  Peak memory per batch: ~{peak_memory_mb:.1f} MB")
    print(f"  Total results size: ~{total_experiments * 0.5 / 1024:.1f} MB")
    
    return {
        'total_experiments': total_experiments,
        'total_batches': total_batches,
        'estimated_time_minutes': total_time_seconds / 60,
        'peak_memory_mb': peak_memory_mb,
        'batch_size': batch_size
    }
