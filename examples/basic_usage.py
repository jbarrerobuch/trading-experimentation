"""
Ejemplo básico de uso del framework Trading Strategy

Este script demuestra cómo usar los módulos para:
1. Cargar datos
2. Calcular indicadores
3. Ejecutar backtests
4. Optimizar estrategias con grid search
5. Visualizar resultados
"""

import sys
import os
from pathlib import Path

# Agregar src/ al path de Python de forma robusta
# Usamos la nueva utilidad de paths si es posible, pero primero necesitamos importar el módulo
# Para importar el módulo, necesitamos agregar src al path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from trading_strategy.utils.paths import get_market_data_dir, get_data_dir
from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    load_strategies_by_name,
    strategy_grid_search,
    visualize_grid_search_results,
    export_best_strategies
)

def main():
    print("=" * 80)
    print("TRADING STRATEGY FRAMEWORK - EJEMPLO BÁSICO")
    print("=" * 80)
    
    # ========== 1. CARGAR DATOS ==========
    print("\n📥 PASO 1: Cargando datos...")
    # Usamos la utilidad de paths para obtener el directorio correcto
    data = load_saved_data('btcusdt', ['1h'])
    
    if not data or '1h' not in data:
        print("❌ No se pudieron cargar datos. Ejecuta primero fetch_ohlcv_data para descargar datos.")
        return
    
    df = data['1h']
    print(f"✓ Datos cargados: {len(df)} candles")
    print(f"  Rango: {df.index.min()} a {df.index.max()}")
    
    # ========== 2. CALCULAR RETORNOS E INDICADORES ==========
    print("\n📊 PASO 2: Calculando retornos e indicadores...")
    df = calculate_returns_and_momentum(
        df, 
        compute_indicators=False  # False para solo retornos (más rápido)
    )
    print(f"✓ Retornos calculados. Columnas disponibles: {len(df.columns)}")
    
    # ========== 3. CARGAR ESTRATEGIAS DESDE YAML ==========
    print("\n📂 PASO 3: Cargando estrategias desde archivos YAML...")
    
    configs = load_strategies_by_name([
        'rsi_basic',
        #'rsi_optimization',
        #'macd_optimization',
        #'williams_r',
        #'rsi_macd_combo'
    ])
    
    if not configs:
        print("❌ No se cargaron estrategias. Verifica que existan archivos en strategies/")
        return
    
    # ========== 4. GRID SEARCH (OPTIMIZACIÓN) ==========
    print("\n🔬 PASO 4: Ejecutando Grid Search...")
    
    grid_results = strategy_grid_search(
        df=df,
        strategy_configs=configs,
        use_mlflow=True,
        ticker='BTCUSDT',
        timeframe='1h',
        experiment_name='RSI_basic'
    )
    
    # ========== 5. ANÁLISIS DE RESULTADOS ==========
    print("\n📊 PASO 5: Analizando resultados...")
    
    if not grid_results.empty:
        print(f"\nTotal estrategias testeadas: {len(grid_results)}")
        print(f"Sharpe promedio: {grid_results['sharpe_ratio'].mean():.3f}")
        print(f"Mejor Sharpe: {grid_results['sharpe_ratio'].max():.3f}")
        
        print("\n🏆 TOP 5 ESTRATEGIAS:")
        print("-" * 80)
        top5 = grid_results.nlargest(5, 'sharpe_ratio')
        for rank, (idx, row) in enumerate(top5.iterrows(), start=1):
            print(f"\n{rank}. {row['strategy_name']}")
            print(f"   Indicador: {row['indicator']}")
            period_val = row.get('period', row.get('lookback', 'N/A'))
            print(f"   Params: period={period_val}, position={row['position_type']}")
            print(f"   Sharpe: {row['sharpe_ratio']:.3f} | Win Rate: {row['win_rate']:.1%} | "
                  f"PF: {row['profit_factor']:.2f} | Trades: {row['n_trades']}")
        
        # ========== 6. VISUALIZACIÓN ==========
        print("\n📈 PASO 6: Generando visualización...")
        data_dir = get_data_dir()
        img_dir = data_dir / 'img'
        img_dir.mkdir(exist_ok=True)
        img_path = img_dir / 'example_grid_search.png'
        
        try:
            visualize_grid_search_results(
                grid_results,
                save_path=str(img_path)
            )
            print(f"✓ Gráfico guardado en: {img_path}")
        except Exception as e:
            print(f"⚠️  Error en visualización: {e}")
        
        # ========== 7. EXPORTACIÓN ==========
        print("\n💾 PASO 7: Exportando mejores estrategias...")
        stats_dir = data_dir / 'stats'
        stats_dir.mkdir(exist_ok=True)
        json_path = stats_dir / 'example_best_strategies.json'
        
        try:
            export_best_strategies(
                grid_results,
                top_n=5,
                save_path=str(json_path)
            )
            print(f"✓ Estrategias exportadas a: {json_path}")
        except Exception as e:
            print(f"⚠️  Error en exportación: {e}")
    
    # ========== FIN ==========
    print("\n" + "=" * 80)
    print("✅ EJEMPLO COMPLETADO")
    print("=" * 80)
    print("\n💡 Próximos pasos:")
    
    # Definir paths por defecto si no se crearon
    if 'img_path' not in locals():
        img_path = "No generado"
    if 'json_path' not in locals():
        json_path = "No generado"

    print(f"1. Ver gráficos en: {img_path}") # type: ignore
    print(f"2. Ver mejores estrategias en: {json_path}") # type: ignore
    print("3. Abrir MLflow UI para explorar experimentos:")
    print(f"   cd {project_root}")
    print("   mlflow ui --backend-store-uri file:./mlruns")
    print("   Abrir: http://localhost:5000")
    print("\n4. Filtrar en MLflow:")
    print("   tags.ticker = 'BTCUSDT' AND tags.timeframe = '1h'")
    print("   metrics.sharpe_ratio > 1.0")


if __name__ == '__main__':
    main()
