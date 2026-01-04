"""
Script para exportar operaciones (trades) de estrategias específicas.
Permite definir configuraciones concretas (no rangos) y obtener el historial de operaciones.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import datetime

# Añadir root al path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    get_strategy_trades,
    load_strategies_by_name
)
from src.trading_strategy.utils.stats import calculate_trade_statistics
from src.trading_strategy.utils.paths import get_project_root

def main():
    # ==========================================
    # 1. CONFIGURACIÓN DE ACTIVOS
    # ==========================================
    tickers = ['ETHUSDT']
    timeframe = '1h'
    
    # ==========================================
    # 2. DEFINICIÓN DE ESTRATEGIAS A EXPORTAR
    # ==========================================
    # Puede ser:
    # a) Nombre del archivo en strategies/ (ej: 'single/rsi_optimization')
    # b) Diccionario con configuración explícita
    
    strategies_input = [
        # --- Opción A: Cargar desde archivo ---
        # 'single/rsi_optimization',
        
        # --- Opción B: Configuración manual ---
        {
            'name': 'ethusd_CCIp14posboth-TRIp25posbothsig15_WEIGHTED1-1.5_B',
            'type': 'combo',
            'indicators_combo': [
                {
                    'indicator': 'cci',
                    'params': {
                        'period': 14, 
                    },
                    'position_type': 'both',
                    'weight': 1.0
                },
                {
                    'indicator': 'trix',
                    'params': {
                        'period': 25,
                        'signal': 15
                    },
                    'position_type': 'both',
                    'weight': 1.5
                }
            ],
            'combination_method': 'WEIGHTED',
            'position_type': 'both'
        },
        #{
        #    'name': 'MACD_Standard',
        #    'type': 'single',
        #    'indicator': 'macd',
        #    'params': {
        #        'fast_period': 12, 
        #        'slow_period': 26, 
        #        'signal_period': 9
        #    },
        #    'position_type': 'both'
        #}
    ]
    
    # Procesar lista mixta
    strategies_to_run = []
    names_to_load = [s for s in strategies_input if isinstance(s, str)]
    
    if names_to_load:
        print(f"📋 Cargando {len(names_to_load)} estrategias desde archivos...")
        loaded_configs = load_strategies_by_name(names_to_load)
        strategies_to_run.extend(loaded_configs)
        
    # Agregar las manuales
    strategies_to_run.extend([s for s in strategies_input if isinstance(s, dict)])
    
    print(f"✓ Total estrategias a procesar: {len(strategies_to_run)}")
    
    # ==========================================
    # 3. PREPARAR DIRECTORIO DE SALIDA
    # ==========================================
    stats_dir = os.path.join(get_project_root(), 'data', 'trades')
    os.makedirs(stats_dir, exist_ok=True)
    print(f"📂 Los trades se guardarán en: {stats_dir}")
    
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # ==========================================
    # 4. PROCESO
    # ==========================================
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"🚀 PROCESANDO {ticker} ({timeframe})")
        print(f"{'='*60}")
        
        # Cargar datos
        data = load_saved_data(ticker=ticker, timeframes=[timeframe])
        if not data or timeframe not in data:
            print(f"❌ No hay datos para {ticker}")
            continue
            
        df = data[timeframe]
        print(f"✓ Datos cargados: {len(df)} velas")
        
        # Calcular retornos (necesario para algunos cálculos internos)
        df = calculate_returns_and_momentum(df, compute_indicators=False)
        
        # Ejecutar cada estrategia
        for strategy in strategies_to_run:
            strat_name = strategy['name']
            print(f"\nRunning: {strat_name}...")
            
            try:
                if strategy['type'] == 'combo':
                    trades_df = get_strategy_trades(
                        df=df,
                        indicator=None,
                        params={},
                        position_type=strategy.get('position_type', 'long'),
                        indicators_combo=strategy['indicators_combo'],
                        combination_method=strategy.get('combination_method', 'AND')
                    )
                else:
                    trades_df = get_strategy_trades(
                        df=df,
                        indicator=strategy['indicator'],
                        params=strategy['params'],
                        position_type=strategy.get('position_type', 'long'),
                        indicators_combo=None
                    )
                
                if not trades_df.empty:
                    # Calcular métricas avanzadas usando el módulo centralizado
                    stats = calculate_trade_statistics(trades_df)
                    
                    print(f"  ✓ Trades generados: {stats['total_trades']}")
                    print(f"  ✓ Win Rate: {stats['win_rate']:.1%}")
                    print(f"  ✓ Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
                    print(f"  ✓ Max Drawdown: {stats['max_drawdown']:.2%}")
                    print(f"  ✓ Total PnL: {stats['total_return']:.2%}")
                    
                    # Guardar CSV
                    filename = f"trades_{ticker}_{timeframe}_{strat_name}.csv"
                    filepath = os.path.join(stats_dir, filename)
                    
                    # Agregar metadatos al CSV (opcional, como columnas constantes)
                    trades_df['ticker'] = ticker
                    trades_df['strategy'] = strat_name
                    
                    trades_df.to_csv(filepath, index=False)
                    print(f"  💾 Guardado en: {filename}")
                else:
                    print("  ⚠️  No se generaron operaciones con esta configuración.")
                    
            except Exception as e:
                print(f"  ❌ Error ejecutando estrategia: {e}")

    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    main()
