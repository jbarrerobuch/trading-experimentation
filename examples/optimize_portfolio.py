"""
Script para optimización de portafolio seleccionando mejores estrategias
desde experimentos de MLflow.
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.trading_strategy.portfolio import PortfolioManager
from src.trading_strategy.data_loader import load_saved_data
from src.trading_strategy.indicators import calculate_returns_and_momentum
from src.trading_strategy.utils.mlflow_utils import (
    setup_mlflow, 
    parse_combo_params, 
    parse_single_params
)

# ========== CONFIGURACIÓN DE BUSQUEDA ==========
# ID del experimento en MLflow (verificar con `mlflow experiments list`)
EXPERIMENT_ID = '1' 

# Filtro de búsqueda (SQL-like syntax de MLflow)
# Ej: "metrics.sqn > 2.5 AND metrics.total_trades > 50"
FILTER_STRING = "metrics.win_rate > 0.2 AND metrics.sqn >= 1.85 AND metrics.profit_factor > 1 AND metrics.calmar_ratio >= 2 AND metrics.max_drawdown >= -0.75"
ORDER_BY = "metrics.sortino_ratio DESC"
# Lista explícita de Run IDs (si se llena, ignora EXPERIMENT_ID y FILTER_STRING)
SPECIFIC_RUN_IDS = [
    # "run_id_1", 
    # "run_id_2"
]

# Métrica para optimizar la combinación de portafolios
# Opciones: 'calmar_ratio', 'sharpe_ratio', 'sortino_ratio', 'total_return', 'cagr', 'max_drawdown'
OPTIMIZATION_METRIC = 'calmar_ratio'

MAX_RUNS = 60

def main():
    # 1. Conectar a MLflow
    if not setup_mlflow():
        print("Error configurando MLflow.")
        return
        
    runs = []
    
    # 2. Obtener Runs
    if SPECIFIC_RUN_IDS:
        print(f"🔍 Cargando {len(SPECIFIC_RUN_IDS)} runs específicos...")
        for run_id in SPECIFIC_RUN_IDS:
            try:
                runs.append(mlflow.get_run(run_id))
            except Exception as e:
                print(f"⚠️  No se encontró run {run_id}: {e}")
    else:
        print(f"🔍 Buscando runs en experimento {EXPERIMENT_ID}...")
        print(f"   Filtro: {FILTER_STRING}")
        
        runs_df = mlflow.search_runs(
            experiment_ids=[EXPERIMENT_ID],
            filter_string=FILTER_STRING,
            max_results=MAX_RUNS,
            order_by=[ORDER_BY]
        )
        
        if runs_df.empty: # pyright: ignore[reportAttributeAccessIssue]
            print("❌ No se encontraron runs que coincidan con los criterios.")
            return

        # Convertir DF de runs a lista de objetos (o diccionarios procesables)
        # Es más fácil iterar sobre el DF search_runs directamente, 
        # pero necesitamos normalizar el formato para el resto del script.
        print(f"✓ Encontrados {len(runs_df)} runs.")
        
    # 3. Agrupar por Ticker/Timeframe 
    # (El portafolio requiere que todas las estrategias operen sobre el mismo asset/timeframe base)
    valid_configs = []
    
    # Si usamos search_runs (DataFrame)
    iter_source = runs if SPECIFIC_RUN_IDS else runs_df.iterrows() # pyright: ignore[reportAttributeAccessIssue, reportPossiblyUnboundVariable]
    
    for item in iter_source:
        if SPECIFIC_RUN_IDS:
            run = item # es objeto Run
            data = run.data.params # pyright: ignore[reportAttributeAccessIssue]
            tags = run.tags # pyright: ignore[reportAttributeAccessIssue]
            metrics = run.data.metrics # pyright: ignore[reportAttributeAccessIssue]
            run_id = run.info.run_id # pyright: ignore[reportAttributeAccessIssue]
        else:
            idx, row = item # es tupla (idx, Series)
            # search_runs aplana params con prefijo 'params.', tags con 'tags.', metrics con 'metrics.'
            run_id = row['run_id']
            
            # Reconstruir dicts sin prefijos para facilitar uso
            data = {k.replace('params.', ''): v for k, v in row.items() if k.startswith('params.')} # pyright: ignore[reportAttributeAccessIssue]
            tags = {k.replace('tags.', ''): v for k, v in row.items() if k.startswith('tags.')} # pyright: ignore[reportAttributeAccessIssue]
            metrics = {k.replace('metrics.', ''): v for k, v in row.items() if k.startswith('metrics.')} # pyright: ignore[reportAttributeAccessIssue]

        # Extraer metadatos clave
        ticker = tags.get('ticker') or data.get('ticker')
        timeframe = tags.get('timeframe') or data.get('timeframe')
        
        if not ticker or not timeframe:
            continue
            
        valid_configs.append({
            'run_id': run_id,
            'ticker': ticker,
            'timeframe': timeframe,
            'params': data,
            'metrics': metrics
        })

    if not valid_configs:
        print("No se pudieron extraer configuraciones válidas de los runs.")
        return

    # Convertir a DF para agrupar
    configs_df = pd.DataFrame(valid_configs)
    
    # Seleccionar el grupo (Ticker, Timeframe) mayoritario
    grp = configs_df.groupby(['ticker', 'timeframe']).size()
    best_group = grp.idxmax() # (Ticker, Timeframe)
    target_ticker, target_timeframe = best_group # pyright: ignore[reportGeneralTypeIssues]
    
    print(f"\n🎯 Grupo seleccionado: {target_ticker} - {target_timeframe} ({grp[best_group]} estrategias)")
    
    # Filtrar solo las de ese grupo
    selected_configs = configs_df[
        (configs_df['ticker'] == target_ticker) & 
        (configs_df['timeframe'] == target_timeframe)
    ]
    
    # 4. Cargar Datos de Mercado
    print(f"📥 Cargando datos de mercado para {target_ticker} {target_timeframe}...")
    try:
        data_dict = load_saved_data(target_ticker, [target_timeframe])
        if data_dict and target_timeframe in data_dict:
            df = data_dict[target_timeframe]
        else:
            raise ValueError(f"No returned data for {target_timeframe}")
            
        # Asegurar retornos
        if 'returns' not in df.columns:
            df = calculate_returns_and_momentum(df, compute_indicators=False)
            
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    # 5. Inicializar Portfolio Manager
    pm = PortfolioManager(df)
    
    print(f"🔄 Procesando estrategias...")
    loaded_count = 0
    names_map = {}
    sqn_map = {}
    
    for _, config in selected_configs.iterrows():
        run_id = config['run_id']
        raw_params = config['params']
        
        # Parsear configuración de estrategia
        strategy_type = raw_params.get('strategy_type', 'single')
        position_type = raw_params.get('position_type', 'long')
        combination_method = raw_params.get('combination_method', 'AND')
        
        indicator = None
        indicators_combo = None
        strategy_params = {}
        
        if strategy_type == 'combo':
            indicators_combo = parse_combo_params(raw_params)
            strat_label = f"Combo_{run_id[:4]}"
        else:
            indicator = raw_params.get('indicator')
            strategy_params = parse_single_params(raw_params)
            strat_label = f"{indicator}_{run_id[:4]}"

        # Short ID para el nombre
        full_name = f"{strat_label}_{run_id}"

        try:
            pm.add_strategy(
                name=full_name,
                indicator=indicator,
                params=strategy_params,
                position_type=position_type,
                indicators_combo=indicators_combo,
                combination_method=combination_method
            )
            
            # Map name for display (include Metric)
            metric_val = config['metrics'].get(OPTIMIZATION_METRIC, 0)
            sqn_val = config['metrics'].get('sqn', 0)
            
            names_map[full_name] = f"{strat_label} (SQN: {sqn_val:.2f})"
            sqn_map[full_name] = sqn_val
            
            loaded_count += 1
            
        except Exception as e:
            print(f"  ⚠️ Error agregando {full_name}: {e}")

    print(f"✓ {loaded_count} estrategias cargadas exitosamente.")
    
    if loaded_count < 2:
        print("❌ Se necesitan al menos 2 estrategias válidas para optimizar.")
        return

    # 5. Output Results match original script flow...
    print("\n--- Matriz de Correlación (Top 5) ---")
    corr = pm.calculate_correlation_matrix()
    print(corr.iloc[:5, :5])
    
    # 6. Optimizar Combinaciones
    print(f"\nBuscando mejores combinaciones de 2 a 3 estrategias (Optimizando por {OPTIMIZATION_METRIC})...")
    best_combos = pm.find_best_combinations(min_k=2, max_k=3, top_n=5, sort_by=OPTIMIZATION_METRIC)
    
    print("\n--- MEJORES COMBINACIONES ---")
    if not best_combos.empty:
        # Reset index to get proper ranking 1..N
        best_combos = best_combos.reset_index(drop=True)
        
        for i, row in best_combos.iterrows():
            strat_list = row['strategies']
            readable_names = [names_map.get(s, s) for s in strat_list]
            
            # Calculate Avg SQN
            avg_sqn = np.mean([sqn_map.get(s, 0) for s in strat_list])

            print(f"\nRank {i+1}:") # pyright: ignore[reportOperatorIssue]
            print(f"Estrategias: {readable_names}")
            print(f"Calmar: {row['calmar_ratio']:.2f} | Sharpe: {row['sharpe_ratio']:.2f} | Avg SQN: {avg_sqn:.2f}")
            print(f"CAGR: {row['cagr']:.2%} | MaxDD: {row['max_drawdown']:.2%}")
            print(f"Retorno Total: {row['total_return']:.2%}")
    else:
        print("No se encontraron combinaciones válidas.")

if __name__ == "__main__":
    main()
