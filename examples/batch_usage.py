"""
Ejemplo de uso del sistema de Batch Grid Search
Para experimentos grandes (50K-200K+)
"""

import sys
import os
import glob
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_strategy import (
    load_saved_data,
    load_strategies_by_name,
    batch_grid_search,
    estimate_batch_requirements,
    calculate_returns_and_momentum,
    get_strategy_trades
)
from src.trading_strategy.utils.paths import get_project_root, get_strategies_dir

# ========== CONSTANTES GLOBALES ==========
# Nombre del experimento en MLflow (común para todos los tickers para facilitar comparación)
EXPERIMENT_NAME = "ETHUSDT_selection"
BATCH_SIZE = 5000


def main():
    # ========== CONFIGURACIÓN ==========
    tickers = [
        'ETHUSDT'
    ]
    timeframe = '1h'
    
    # ========== 1. CARGAR ESTRATEGIAS (Común para todos) ==========
    print("\n📋 Cargando configuraciones de estrategias...")
    
    # Cargar todas las estrategias de la carpeta 'single'
    strategies_root = get_strategies_dir()
    combo_dir = os.path.join(strategies_root, '')
    
    strategy_files = glob.glob(os.path.join(combo_dir, '*.yaml'))
    strategy_names = [f"ETH_cci-trix/{os.path.splitext(os.path.basename(f))[0]}" for f in strategy_files]
    strategy_names.sort()
    
    print(f"  Estrategias encontradas en: {len(strategy_names)}")
    
    configs = load_strategies_by_name(strategy_names)
    
    print(f"✓ {len(configs)} configuraciones cargadas")
    
    # ========== 2. ESTIMAR RECURSOS ==========
    print("\n🔍 Estimando recursos necesarios...")
    estimate_batch_requirements(configs, batch_size=BATCH_SIZE)
    
    # ========== 3. PREPARAR DIRECTORIOS ==========
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_checkpoints')
    print(f"📂 Checkpoints se guardarán en: {checkpoint_dir}")
    
    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    input(f"⏸️  Presiona ENTER para iniciar batch grid search para {tickers} (o Ctrl+C para cancelar)...")
    
    # ========== 4. PROCESAR CADA ACTIVO ==========
    for ticker in tickers:
        print("\n" + "="*80)
        print(f"🚀 PROCESANDO {ticker} ({timeframe})")
        print("="*80)
        
        # Cargar datos
        print(f"📥 Cargando datos para {ticker}...")
        data = load_saved_data(
            ticker=ticker,
            timeframes=[timeframe]
        )
        
        if not data or timeframe not in data:
            print(f"❌ No se encontraron datos para {ticker} {timeframe}")
            continue

        df = data[timeframe]
        print(f"✓ Datos cargados: {len(df)} velas")
        print(f"  Periodo: {df.index[0]} - {df.index[-1]}")
        
        # Calcular retornos
        print("\n📊 Calculando retornos e indicadores base...")
        df = calculate_returns_and_momentum(
            df,
            compute_indicators=False,
            lookforward_periods=[1]
        )
        print("✓ Retornos pre-calculados")
        
        # Ejecutar Batch Search
        print(f"🧪 Experimento MLflow: {EXPERIMENT_NAME}")
        
        # Obtener rango de fechas para el nombre del archivo
        start_date = df.index[0].strftime("%Y%m%d")
        end_date = df.index[-1].strftime("%Y%m%d")
        
        # Preparar directorio de salida
        results_dir = os.path.join(get_project_root(), 'data', 'final_results')
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f"batch_results_{ticker}_{timeframe}_{start_date}-{end_date}_summary_{date_str}.csv")
        
        results = batch_grid_search(
            df=df,
            strategy_configs=configs,
            batch_size=BATCH_SIZE,          # 10K experimentos por batch
            use_mlflow=True,            # Registrar en MLflow
            ticker=ticker,
            timeframe=timeframe,
            experiment_name=EXPERIMENT_NAME,
            save_checkpoints=True,      # Guardar checkpoints intermedios
            checkpoint_dir=checkpoint_dir,
            output_file=output_file     # Guardar directamente en final_results
        )
        
        # ========== 5. ANALIZAR RESULTADOS ==========
        print(f"\n📊 Análisis de resultados para {ticker}...")
        
        if not results.empty:
            print(f"\nTotal resultados: {len(results):,}")
            print(f"Mejor Sharpe: {results['sharpe_ratio'].max():.3f}")
            print(f"Peor Sharpe: {results['sharpe_ratio'].min():.3f}")
            print(f"Promedio Sharpe: {results['sharpe_ratio'].mean():.3f}")
            
            # Top 10 estrategias
            print(f"\n🏆 TOP 10 ESTRATEGIAS ({ticker}):")
            print("=" * 100)
            top10 = results.nlargest(10, 'sharpe_ratio')
            
            # Directorios de salida
            trades_dir = os.path.join(get_project_root(), 'data', 'trades')
            results_dir = os.path.join(get_project_root(), 'data', 'final_results')
            os.makedirs(trades_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                print(f"\n{idx}. {row['strategy_name']} - Sharpe: {row['sharpe_ratio']:.3f}")
                print(f"   Win Rate: {row['win_rate']:.1%} | Total Return: {row['total_return']:.2%}")
                print(f"   Max DD: {row['max_drawdown']:.2%} | Calmar: {row['calmar_ratio']:.2f}")
                
                strategy_type = row.get('strategy_type', 'single')
                position_type = row.get('position_type', 'long')
                combination_method = row.get('combination_method', 'AND')
                
                if strategy_type == 'combo':
                    print(f"   Indicators: {row.get('ind1_name', '?')}, {row.get('ind2_name', '?')}")
                    print(f"   Method: {combination_method}")
                    
                    # Reconstruir params para combo
                    indicator = None
                    params = {}
                    indicators_combo = []
                    n_indicators = int(row.get('n_indicators', 0))
                    
                    for i in range(1, n_indicators + 1):
                        ind_name = row.get(f'ind{i}_name')
                        ind_params = {}
                        prefix = f'ind{i}_'
                        for col in row.index:
                            if col.startswith(prefix) and col != f'{prefix}name' and not pd.isna(row[col]):
                                param_key = col[len(prefix):]
                                ind_params[param_key] = row[col]
                        
                        indicators_combo.append({
                            'indicator': ind_name,
                            'params': ind_params
                        })
                        
                else:
                    print(f"   Indicator: {row.get('indicator', 'N/A')}")
                    print(f"   Params: period={row.get('period', 'N/A')}, "
                          f"oversold={row.get('oversold', 'N/A')}, "
                          f"overbought={row.get('overbought', 'N/A')}")
                    
                    # Reconstruir params para single
                    indicator = row.get('indicator')
                    indicators_combo = None
                    
                    exclude_cols = {
                        'ticker', 'timeframe', 'strategy_name', 'strategy_type', 
                        'indicator', 'position_type', 'combination_method', 'n_indicators',
                        'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown',
                        '_batch_id', '_session_id', 'n_trades', 'n_transactions',
                        'hit_rate', 'sortino_ratio', 'calmar_ratio', 'profit_factor',
                        'avg_win', 'avg_loss', 'risk_reward_ratio', 'best_trade',
                        'worst_trade', 'avg_return_per_trade', 'volatility', 'sqn',
                        'kelly_criterion', 'expectancy', 'net_profit_usd', 'final_equity',
                        'max_drawdown_usd', 'total_costs_pct'
                    }
                    
                    params = {}
                    for col in row.index:
                        if col not in exclude_cols and not pd.isna(row[col]):
                             params[col] = row[col]

                # Exportar trades para el Top 5 (DESACTIVADO POR DEFECTO)
                # Para generar trades, usar el script examples/generate_artifacts_from_mlflow.py
                if False and idx <= 5:
                    try:
                        trades_df = get_strategy_trades(
                            df=df,
                            indicator=indicator,
                            params=params,
                            position_type=position_type,
                            indicators_combo=indicators_combo,
                            combination_method=combination_method
                        )
                        
                        if not trades_df.empty:
                            trades_file = os.path.join(trades_dir, f"trades_{ticker}_{timeframe}_rank{idx}_{date_str}.csv")
                            trades_df.to_csv(trades_file, index=False)
                            print(f"   💾 Trades exportados: {os.path.basename(trades_file)}")
                    except Exception as e:
                        print(f"   ⚠️ Error exportando trades: {e}")
            
            # El archivo de resultados ya fue guardado por batch_grid_search en output_file
            
        else:
            print(f"⚠️  No se obtuvieron resultados válidos para {ticker}")

    print("\n✅ Proceso completado para todos los activos")


if __name__ == "__main__":
    main()
