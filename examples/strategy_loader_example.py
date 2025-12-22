"""
Ejemplo de uso del sistema de carga de estrategias desde YAML

Demuestra cómo:
1. Cargar todas las estrategias desde strategies/
2. Cargar estrategias específicas por nombre
3. Ver información de estrategias disponibles
4. Ejecutar grid search con estrategias cargadas
"""

import sys
import os

# Agregar src/ al path de Python
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    load_all_strategies,
    load_strategies_by_name,
    get_strategy_info,
    strategy_grid_search
)


def main():
    print("=" * 80)
    print("EJEMPLO: CARGA DE ESTRATEGIAS DESDE YAML")
    print("=" * 80)
    
    # ========== 1. LISTAR ESTRATEGIAS DISPONIBLES ==========
    print("\n📋 PASO 1: Información de estrategias disponibles")
    print("-" * 80)
    
    info = get_strategy_info()
    print(f"\nTotal estrategias: {info['total']}")
    
    if info['individual']:
        print(f"\n📊 Estrategias Individuales ({len(info['individual'])}):")
        for strat in info['individual']:
            print(f"  - {strat['name']} ({strat['indicator']}) → {strat['file']}")
    
    if info['combo']:
        print(f"\n🔗 Estrategias Combo ({len(info['combo'])}):")
        for strat in info['combo']:
            print(f"  - {strat['name']} ({strat['n_indicators']} indicadores) → {strat['file']}")
    
    # ========== 2. CARGAR TODAS LAS ESTRATEGIAS ==========
    print("\n\n📂 PASO 2: Cargando TODAS las estrategias")
    print("-" * 80)
    
    all_strategies = load_all_strategies()
    
    # ========== 3. CARGAR ESTRATEGIAS ESPECÍFICAS ==========
    print("\n\n🎯 PASO 3: Cargando estrategias específicas")
    print("-" * 80)
    
    specific_strategies = load_strategies_by_name([
        'rsi_optimization',
        'macd_optimization'
    ])
    
    # ========== 4. EJECUTAR GRID SEARCH (OPCIONAL) ==========
    print("\n\n🔬 PASO 4: Ejecutar grid search con estrategias cargadas")
    print("-" * 80)
    
    run_grid_search = input("¿Ejecutar grid search? (y/n): ").lower().strip() == 'y'
    
    if run_grid_search:
        # Cargar datos
        print("\n📥 Cargando datos...")
        data_path = os.path.join(project_root, 'data', 'market')
        data = load_saved_data('btcusdt', ['1h'], data_path=data_path)
        
        if not data or '1h' not in data:
            print("❌ No se pudieron cargar datos")
            return
        
        df = data['1h']
        print(f"✓ Datos cargados: {len(df)} candles")
        
        # Calcular retornos
        print("📊 Calculando retornos...")
        df = calculate_returns_and_momentum(df, compute_indicators=False)
        print(f"✓ Retornos calculados")
        
        # Ejecutar grid search
        print("\n🔬 Ejecutando Grid Search...")
        results = strategy_grid_search(
            df=df,
            strategy_configs=specific_strategies,  # Usar estrategias cargadas
            use_mlflow=True,
            ticker='BTCUSDT',
            timeframe='1h',
            experiment_name='yaml_strategies_test'
        )
        
        if not results.empty:
            print(f"\n✓ Grid Search completado: {len(results)} resultados")
            print(f"  Mejor Sharpe: {results['sharpe_ratio'].max():.3f}")
            
            print("\n🏆 TOP 3 ESTRATEGIAS:")
            top3 = results.nlargest(3, 'sharpe_ratio')
            for rank, (idx, row) in enumerate(top3.iterrows(), start=1):
                print(f"{rank}. {row['strategy_name']} - Sharpe: {row['sharpe_ratio']:.3f}")
        else:
            print("⚠️  No se generaron resultados")
    
    # ========== FIN ==========
    print("\n" + "=" * 80)
    print("✅ EJEMPLO COMPLETADO")
    print("=" * 80)
    print("\n💡 Cómo usar en tus scripts:")
    print("""
    # Opción 1: Cargar todas las estrategias
    from trading_strategy import load_all_strategies, strategy_grid_search
    configs = load_all_strategies()
    results = strategy_grid_search(df, configs)
    
    # Opción 2: Cargar estrategias específicas
    from trading_strategy import load_strategies_by_name
    configs = load_strategies_by_name(['rsi_optimization', 'macd_optimization'])
    results = strategy_grid_search(df, configs)
    
    # Opción 3: Ver qué estrategias hay disponibles
    from trading_strategy import get_strategy_info
    info = get_strategy_info()
    print(info)
    """)


if __name__ == '__main__':
    main()
