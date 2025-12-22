"""
Ejemplo de uso del sistema de Batch Grid Search
Para experimentos grandes (50K-200K+)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_strategy import (
    load_saved_data,
    load_all_strategies,
    batch_grid_search,
    estimate_batch_requirements,
    calculate_returns_and_momentum
)


def main():
    # ========== 1. CARGAR DATOS ==========
    print("📥 Cargando datos...")
    df = load_saved_data(
        ticker='BTCUSDT',
        timeframe='1h',
        start_date='2017-08-18',
        end_date='2025-11-11'
    )
    
    print(f"✓ Datos cargados: {len(df)} velas")
    print(f"  Periodo: {df.index[0]} - {df.index[-1]}")
    
    # ========== 1.5 CALCULAR RETORNOS ==========
    print("\n📊 Calculando retornos e indicadores base...")
    # compute_indicators=False para solo calcular retornos y future_ret+N
    # Esto optimiza el grid search evitando recálculos
    df = calculate_returns_and_momentum(df, compute_indicators=False)
    print("✓ Retornos pre-calculados")
    
    # ========== 2. CARGAR ESTRATEGIAS ==========
    print("\n📋 Cargando configuraciones de estrategias...")
    
    # Opción A: Cargar todas las estrategias
    configs = load_all_strategies()
    
    # Opción B: Cargar solo algunas
    # configs = load_strategies_by_name(['rsi_optimization', 'macd_optimization'])
    
    print(f"✓ {len(configs)} configuraciones cargadas")
    
    # ========== 3. ESTIMAR RECURSOS ==========
    print("\n🔍 Estimando recursos necesarios...")
    
    # Probar diferentes tamaños de batch
    batch_sizes = [5000, 10000, 20000]
    
    for batch_size in batch_sizes:
        print(f"\n--- Con batch_size = {batch_size:,} ---")
        estimate_batch_requirements(configs, batch_size=batch_size)
    
    # ========== 4. EJECUTAR BATCH GRID SEARCH ==========
    print("\n" + "="*80)
    input("⏸️  Presiona ENTER para iniciar batch grid search (o Ctrl+C para cancelar)...")
    
    results = batch_grid_search(
        df=df,
        strategy_configs=configs,
        batch_size=10000,          # 10K experimentos por batch
        use_mlflow=True,            # Registrar en MLflow
        ticker='BTCUSDT',
        timeframe='1h',
        experiment_name='btc_1h_batch_test',
        save_checkpoints=True       # Guardar checkpoints intermedios
    )
    
    # ========== 5. ANALIZAR RESULTADOS ==========
    print("\n📊 Análisis de resultados...")
    
    if not results.empty:
        print(f"\nTotal resultados: {len(results):,}")
        print(f"Mejor Sharpe: {results['sharpe_ratio'].max():.3f}")
        print(f"Peor Sharpe: {results['sharpe_ratio'].min():.3f}")
        print(f"Promedio Sharpe: {results['sharpe_ratio'].mean():.3f}")
        
        # Top 10 estrategias
        print(f"\n🏆 TOP 10 ESTRATEGIAS:")
        print("=" * 100)
        top10 = results.nlargest(10, 'sharpe_ratio')
        
        for idx, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"\n{idx}. {row['strategy_name']} - Sharpe: {row['sharpe_ratio']:.3f}")
            print(f"   Win Rate: {row['win_rate']:.1%} | Total Return: {row['total_return']:.2%}")
            print(f"   Max DD: {row['max_drawdown']:.2%} | Calmar: {row['calmar_ratio']:.2f}")
            
            if row.get('strategy_type') == 'combo':
                print(f"   Indicators: {row.get('ind1_name', '?')}, {row.get('ind2_name', '?')}")
                print(f"   Method: {row['combination_method']}")
            else:
                print(f"   Indicator: {row.get('indicator', 'N/A')}")
                print(f"   Params: period={row.get('period', 'N/A')}, "
                      f"oversold={row.get('oversold', 'N/A')}, "
                      f"overbought={row.get('overbought', 'N/A')}")
        
        # Análisis por tipo de estrategia
        print(f"\n📈 ANÁLISIS POR TIPO:")
        print("=" * 100)
        
        by_type = results.groupby('strategy_type').agg({
            'sharpe_ratio': ['count', 'mean', 'max'],
            'win_rate': 'mean',
            'total_return': 'mean'
        }).round(3)
        print(by_type)
        
        # Análisis por batch (ver si hay degradación)
        if '_batch_id' in results.columns:
            print(f"\n📦 ANÁLISIS POR BATCH:")
            print("=" * 100)
            
            by_batch = results.groupby('_batch_id').agg({
                'sharpe_ratio': ['count', 'mean', 'max'],
                'win_rate': 'mean'
            }).round(3)
            print(by_batch)
        
        print(f"\n✅ Proceso completado")
        print(f"💾 Resultados guardados en: batch_results_btc_1h_batch_test_*_FINAL.csv")
        print(f"💾 Checkpoints en: checkpoint_btc_1h_batch_test_*.csv")
        
    else:
        print("⚠️  No se obtuvieron resultados válidos")


if __name__ == "__main__":
    main()
