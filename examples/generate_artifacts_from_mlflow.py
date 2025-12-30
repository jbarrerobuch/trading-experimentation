"""
Script para generar artefactos (trades, gráficos) a partir de runs de MLflow seleccionados.
Permite analizar runs interesantes sin ralentizar el grid search masivo.
"""

import sys
import os
import re
import pandas as pd
import mlflow
from pathlib import Path

# Agregar src/ al path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    get_strategy_trades
)
from trading_strategy.utils.mlflow_viz import create_interactive_trade_chart
from trading_strategy.utils.paths import get_project_root

# ========== CONFIGURACIÓN ==========
MLFLOW_DB_URI = f"sqlite:///{project_root}/mlflow.db"
OUTPUT_DIR = project_root / 'data' / 'artifacts_generated'

# Filtros para seleccionar runs interesantes
FILTER_STRING = (
    "metrics.sharpe_ratio > 1.5 "
    "AND tags.ticker = 'BTCUSDT' "
    "AND tags.timeframe = '1h'"
)
# Opcional: Lista de Run IDs específicos (si se define, ignora FILTER_STRING)
SPECIFIC_RUN_IDS = [] 

# Máximo de runs a procesar (para evitar procesar miles por error)
MAX_RUNS = 10

def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_DB_URI)
    print(f"🔌 Conectado a MLflow: {MLFLOW_DB_URI}")

def parse_combo_params(params):
    """Reconstruye la estructura de indicadores para estrategias combo desde params planos de MLflow"""
    indicators_combo = []
    
    # Determinar cuántos indicadores hay
    n_indicators = int(params.get('n_indicators', 0))
    
    for i in range(1, n_indicators + 1):
        ind_name = params.get(f'ind{i}_name')
        if not ind_name:
            continue
            
        ind_params = {}
        prefix = f'ind{i}_'
        
        for key, value in params.items():
            if key.startswith(prefix) and key != f'{prefix}name':
                param_name = key[len(prefix):]
                # Intentar convertir a número si es posible
                try:
                    if '.' in value:
                        ind_params[param_name] = float(value)
                    else:
                        ind_params[param_name] = int(value)
                except ValueError:
                    ind_params[param_name] = value
        
        indicators_combo.append({
            'indicator': ind_name,
            'params': ind_params
        })
        
    return indicators_combo

def parse_single_params(params):
    """Limpia y convierte parámetros para estrategias individuales"""
    clean_params = {}
    exclude_keys = {
        'strategy_name', 'strategy_type', 'indicator', 'position_type', 
        'train_split', 'ticker', 'timeframe', 'git_commit', 'session_id'
    }
    
    for k, v in params.items():
        if k not in exclude_keys and not k.startswith('ind'):
            try:
                if '.' in v:
                    clean_params[k] = float(v)
                else:
                    clean_params[k] = int(v)
            except ValueError:
                clean_params[k] = v
                
    return clean_params

def main():
    setup_mlflow()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Buscar runs
    if SPECIFIC_RUN_IDS:
        runs = []
        for run_id in SPECIFIC_RUN_IDS:
            try:
                runs.append(mlflow.get_run(run_id))
            except Exception as e:
                print(f"⚠️  No se encontró run {run_id}: {e}")
    else:
        print(f"🔍 Buscando runs con filtro: {FILTER_STRING}")
        runs = mlflow.search_runs(
            filter_string=FILTER_STRING,
            max_results=MAX_RUNS,
            order_by=["metrics.sharpe_ratio DESC"]
        )
        # Convertir DataFrame de runs a lista de objetos Run si es necesario, 
        # pero search_runs devuelve DataFrame. Iteraremos sobre el DF.
    
    if isinstance(runs, pd.DataFrame):
        if runs.empty:
            print("❌ No se encontraron runs que coincidan con el filtro.")
            return
        print(f"✓ Encontrados {len(runs)} runs.")
        runs_iter = runs.iterrows()
    else:
        # Si es lista de objetos Run (caso SPECIFIC_RUN_IDS)
        if not runs:
            print("❌ No hay runs para procesar.")
            return
        print(f"✓ Procesando {len(runs)} runs específicos.")
        # Convertir a formato similar para iterar
        runs_data = []
        for run in runs:
            data = run.data.params
            data.update(run.data.metrics)
            data.update(run.data.tags)
            data['run_id'] = run.info.run_id
            runs_data.append(data)
        runs_iter = pd.DataFrame(runs_data).iterrows()

    # Cache de datos para no recargar si es el mismo ticker/tf
    data_cache = {}

    for idx, (_, row) in enumerate(runs_iter):
        run_id = row.get('run_id')
        print(f"\nProcessing Run {idx+1}/{len(runs) if isinstance(runs, list) else len(runs)}: {run_id}")
        
        # Extraer info básica
        ticker = row.get('tags.ticker', row.get('ticker'))
        timeframe = row.get('tags.timeframe', row.get('timeframe'))
        strategy_type = row.get('params.strategy_type', row.get('strategy_type'))
        strategy_name = row.get('params.strategy_name', row.get('strategy_name'))
        
        if not ticker or not timeframe:
            print("⚠️  Falta ticker o timeframe en tags. Saltando.")
            continue
            
        # Cargar datos si es necesario
        data_key = f"{ticker}_{timeframe}"
        if data_key not in data_cache:
            print(f"📥 Cargando datos para {ticker} {timeframe}...")
            data = load_saved_data(ticker.lower(), [timeframe])
            if not data or timeframe not in data:
                print(f"❌ No se pudieron cargar datos para {ticker}.")
                continue
            
            df = data[timeframe]
            df = calculate_returns_and_momentum(df, compute_indicators=False, lookforward_periods=[1])
            data_cache[data_key] = df
        
        df = data_cache[data_key]
        
        # Reconstruir configuración
        # Re-extraer params limpios del row completo asumiendo prefijos de MLflow
        raw_params = {}
        for col in row.index:
            if col.startswith('params.'):
                raw_params[col.replace('params.', '')] = row[col]
        
        position_type = raw_params.get('position_type', 'long')
        combination_method = raw_params.get('combination_method', 'AND')
        
        indicator = None
        indicators_combo = None
        strategy_params = {}
        
        if strategy_type == 'combo':
            indicators_combo = parse_combo_params(raw_params)
            print(f"   Estrategia Combo: {len(indicators_combo)} indicadores")
        else:
            indicator = raw_params.get('indicator')
            strategy_params = parse_single_params(raw_params)
            print(f"   Estrategia Single: {indicator}")

        # Generar Trades
        try:
            print("   Generando trades...")
            trades_df = get_strategy_trades(
                df=df,
                indicator=indicator,
                params=strategy_params,
                position_type=position_type,
                indicators_combo=indicators_combo,
                combination_method=combination_method
            )
            
            if not trades_df.empty:
                # Guardar CSV
                safe_name = re.sub(r'[^\w\-]', '_', f"{strategy_name}_{run_id}")
                csv_path = OUTPUT_DIR / f"trades_{safe_name}.csv"
                trades_df.to_csv(csv_path, index=False)
                print(f"   💾 Trades guardados: {csv_path.name}")
                
                # Generar Gráfico
                print("   Generando gráfico interactivo...")
                html_path = OUTPUT_DIR / f"viz_{safe_name}.html"
                
                viz_indicators = indicators_combo if strategy_type == 'combo' else [{'indicator': indicator, 'params': strategy_params}]
                
                create_interactive_trade_chart(
                    df=df,
                    trades_df=trades_df,
                    title=f"{strategy_name} - {ticker} ({timeframe}) - Sharpe: {row.get('metrics.sharpe_ratio', 0):.2f}",
                    filename=str(html_path),
                    indicators=viz_indicators
                )
                print(f"   📈 Gráfico guardado: {html_path.name}")
                
                # Opcional: Loguear como artefacto al run original en MLflow
                # print("   Subiendo artefactos a MLflow...")
                # with mlflow.start_run(run_id=run_id):
                #     mlflow.log_artifact(str(csv_path), artifact_path="generated_artifacts")
                #     mlflow.log_artifact(str(html_path), artifact_path="generated_artifacts")
                
            else:
                print("   ⚠️  No se generaron trades (estrategia no operó).")
                
        except Exception as e:
            print(f"   ❌ Error procesando run: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    main()
