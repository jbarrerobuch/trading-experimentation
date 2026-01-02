"""
Script para generar artefactos (trades, gráficos) a partir de runs de MLflow seleccionados.
Permite analizar runs interesantes sin ralentizar el grid search masivo.
"""

import sys
import os
import re
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Fix unicode output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8') # pyright: ignore[reportAttributeAccessIssue]

print("Iniciando script de generación de artefactos...")

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
from trading_strategy.utils.mlflow_utils import (
    setup_mlflow,
    parse_combo_params,
    parse_single_params
)

# ========== CONFIGURACIÓN ==========
# Los artefactos se guardarán en data/mlruns/{run_id}/{plots|trades}

# ID de Experimento para filtrar (None para buscar en todos)
EXPERIMENT_ID = '2'

# Filtros para seleccionar runs interesantes
FILTER_STRING = ""

# Opcional: Lista de Run IDs específicos (si se define, ignora FILTER_STRING)
SPECIFIC_RUN_IDS = ["8c6ffdac38cb411d877e379fa605f1f2"] 

# Máximo de runs a procesar (para evitar procesar miles por error)
MAX_RUNS = 10

def main():
    # Validar configuración inicial
    if not EXPERIMENT_ID:
        print("❌ Error: Debes especificar un EXPERIMENT_ID válido o una lista de SPECIFIC_RUN_IDS.")
        return

    # Configurar MLflow
    if not setup_mlflow():
        return
    
    client = MlflowClient()
    
    # Buscar runs
    if SPECIFIC_RUN_IDS:
        runs = []
        for run_id in SPECIFIC_RUN_IDS:
            try:
                runs.append(mlflow.get_run(run_id))
            except Exception as e:
                print(f"⚠️  No se encontró run {run_id}: {e}")
    else:
        experiment_ids = [str(EXPERIMENT_ID)]
        print(f"🔍 Buscando en experimento específico: {experiment_ids}")
        
        print(f"🔍 Buscando runs con filtro: {FILTER_STRING}")
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=FILTER_STRING,
            max_results=MAX_RUNS,
            order_by=["metrics.sharpe_ratio DESC"]
        )
    
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
            data = {}
            # Agregar prefijos para simular estructura de search_runs
            for k, v in run.data.params.items():
                data[f'params.{k}'] = v
            for k, v in run.data.metrics.items():
                data[f'metrics.{k}'] = v
            for k, v in run.data.tags.items():
                data[f'tags.{k}'] = v
                
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
                # Definir rutas de salida basadas en el run_id
                run_root = project_root / 'data' / 'mlruns' / run_id # pyright: ignore[reportOperatorIssue]
                trades_dir = run_root / 'trades'
                plots_dir = run_root / 'plots'
                
                trades_dir.mkdir(parents=True, exist_ok=True)
                plots_dir.mkdir(parents=True, exist_ok=True)

                # Guardar CSV
                safe_name = re.sub(r'[^\w\-]', '_', f"{strategy_name}")
                csv_path = trades_dir / f"trades_{safe_name}.csv"
                trades_df.to_csv(csv_path, index=False)
                print(f"   💾 Trades guardados: {csv_path}")
                
                # Generar Gráfico
                print("   Generando gráfico interactivo...")
                html_path = plots_dir / f"viz_{safe_name}.html"
                
                viz_indicators = indicators_combo if strategy_type == 'combo' else [{'indicator': indicator, 'params': strategy_params}]
                
                create_interactive_trade_chart(
                    df=df,
                    trades_df=trades_df,
                    title=f"{strategy_name} - {ticker} ({timeframe}) - Sharpe: {row.get('metrics.sharpe_ratio', 0):.2f}",
                    filename=str(html_path),
                    indicators=viz_indicators
                )
                print(f"   📈 Gráfico guardado: {html_path}")
                
                # Loguear como artefacto al run original en MLflow usando MlflowClient
                print("   Subiendo artefactos a MLflow...")
                try:
                    client.log_artifact(run_id, str(csv_path), artifact_path="generated_artifacts")
                    client.log_artifact(run_id, str(html_path), artifact_path="generated_artifacts")
                    print("   ✓ Artefactos subidos a MLflow")
                except Exception as e:
                    print(f"   ⚠️  Error subiendo a MLflow: {e}")
                
            else:
                print("   ⚠️  No se generaron trades (estrategia no operó).")
                
        except Exception as e:
            print(f"   ❌ Error procesando run: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    main()
