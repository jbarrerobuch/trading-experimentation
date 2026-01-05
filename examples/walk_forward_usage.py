"""
Ejemplo de uso de Walk-Forward Optimization (WFA)
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Agregar src/ al path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    walk_forward_optimization
)

def main():
    print("=" * 80)
    print("WALK-FORWARD OPTIMIZATION EXAMPLE")
    print("=" * 80)
    
    # 1. Cargar datos
    ticker = 'ETHUSDT'
    timeframe = '1h'
    print(f"\n📥 Cargando datos para {ticker} {timeframe}...")
    
    data = load_saved_data(ticker, [timeframe])
    if not data or timeframe not in data:
        print("❌ No se encontraron datos. Ejecuta fetch_ohlcv_data primero.")
        return
        
    df = data[timeframe]
    print(f"✓ Datos cargados: {len(df)} velas")
    
    # 2. Calcular retornos (necesario para backtest)
    print("\n📊 Calculando retornos...")
    df = calculate_returns_and_momentum(df, compute_indicators=False)
    
    # 3. Definir espacio de búsqueda (Configuración)
    # Ejemplo simple con RSI
    strategy_configs = [{
        'name': 'RSI_WFA_Test',
        'indicator': 'rsi',
        'params_grid': {
            'period': [10, 14, 20],
            'overbought': [70, 75, 80],
            'oversold': [20, 25, 30],
            'position_type': ['long', 'short']
        }
    }]
    
    # 4. Configurar WFA
    # Usaremos ventanas pequeñas para que el ejemplo corra rápido
    # En producción: Window ~3-6 meses, Step ~1 mes
    # Aquí: Window ~1 mes (720h), Step ~1 semana (168h)
    window_size = 720 * 2  # 2 meses entrenamiento
    step_size = 168 * 2    # 2 semanas prueba
    
    print(f"\n🔬 Configuración WFA:")
    print(f"   Window (Train): {window_size} velas")
    print(f"   Step (Test): {step_size} velas")
    print(f"   Estrategias a probar: {strategy_configs[0]['name']}")
    
    # 5. Ejecutar WFA
    results = walk_forward_optimization(
        df=df.tail(5000), # Usar solo las últimas 5000 velas para demo rápida
        strategy_configs=strategy_configs,
        window_size=window_size,
        step_size=step_size,
        metric='sharpe_ratio',
        anchored=False, # Rolling window
        use_mlflow=False, # Desactivado para demo simple
        ticker=ticker,
        timeframe=timeframe
    )
    
    # 6. Mostrar resultados
    metrics = results['metrics']
    if metrics:
        print("\n🏆 RESULTADOS FINALES WFA:")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        
        # Mostrar historial de cambios de parámetros
        print("\n📜 Historial de Re-optimización (últimos 5 pasos):")
        history = results['history']
        print(history[['step', 'test_start_date', 'best_strategy', 'is_score', 'oos_pnl']].tail())
        
        # Guardar resultados
        output_dir = project_root / 'data' / 'stats'
        os.makedirs(output_dir, exist_ok=True)
        history.to_csv(output_dir / 'wfa_example_history.csv', index=False)
        print(f"\n💾 Historial guardado en data/stats/wfa_example_history.csv")
    else:
        print("\n⚠️ No se generaron trades en el periodo OOS.")

if __name__ == "__main__":
    main()
