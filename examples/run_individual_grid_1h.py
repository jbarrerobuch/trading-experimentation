"""
Grid search de la colección 'individual' (28 indicadores, 3.306 combos)
sobre los 6 tickers disponibles en timeframe 1h.
"""

import sys
import os
import glob
import time
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_strategy import (
    load_saved_data,
    load_strategies_by_name,
    batch_grid_search,
    estimate_batch_requirements,
    calculate_returns_and_momentum,
)
from src.trading_strategy.utils.paths import get_project_root, get_strategies_dir

EXPERIMENT_NAME = "individual_grid_1h"
BATCH_SIZE = 10000
STRATEGY_FOLDER = 'individual'
TICKERS = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'SOLUSDT', 'UNIUSDT', 'ZECUSDT']
TIMEFRAME = '1h'


def main():
    print("\n📋 Cargando configuraciones de estrategias...")

    strategies_root = get_strategies_dir()
    strategy_dir = os.path.join(strategies_root, STRATEGY_FOLDER)

    strategy_files = glob.glob(os.path.join(strategy_dir, '*.yaml'))
    strategy_names = [f"{STRATEGY_FOLDER}/{os.path.splitext(os.path.basename(f))[0]}" for f in strategy_files]
    strategy_names.sort()

    print(f"  Estrategias encontradas: {len(strategy_names)}")

    configs = load_strategies_by_name(strategy_names)

    print(f"✓ {len(configs)} configuraciones cargadas")

    print("\n🔍 Estimando recursos necesarios...")
    estimate_batch_requirements(configs, batch_size=BATCH_SIZE)

    checkpoint_dir = os.path.join(get_project_root(), 'data', 'checkpoints_individual_1h')
    results_dir = os.path.join(get_project_root(), 'data', 'results_individual_1h')
    final_results_dir = os.path.join(get_project_root(), 'data', 'final_results')
    os.makedirs(final_results_dir, exist_ok=True)

    print(f"📂 Checkpoints: {checkpoint_dir}")
    print(f"📂 Parquet + manifest: {results_dir}")

    import datetime
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    overall_start = time.time()

    for ticker in TICKERS:
        print("\n" + "=" * 80)
        print(f"🚀 PROCESANDO {ticker} ({TIMEFRAME})")
        print("=" * 80)

        ticker_start = time.time()

        print(f"📥 Cargando datos para {ticker}...")
        data = load_saved_data(ticker=ticker, timeframes=[TIMEFRAME])

        if not data or TIMEFRAME not in data:
            print(f"❌ No se encontraron datos para {ticker} {TIMEFRAME}")
            continue

        df = data[TIMEFRAME]
        print(f"✓ Datos cargados: {len(df)} velas")
        print(f"  Periodo: {df.index[0]} - {df.index[-1]}")

        print("\n📊 Calculando retornos e indicadores base...")
        df = calculate_returns_and_momentum(df, compute_indicators=False, lookforward_periods=[1])
        print("✓ Retornos pre-calculados")

        start_date = df.index[0].strftime("%Y%m%d")
        end_date = df.index[-1].strftime("%Y%m%d")

        output_file = os.path.join(
            final_results_dir,
            f"batch_results_individual_{ticker}_{TIMEFRAME}_{start_date}-{end_date}_{date_str}.csv"
        )

        results = batch_grid_search(
            df=df,
            strategy_configs=configs,
            batch_size=BATCH_SIZE,
            ticker=ticker,
            timeframe=TIMEFRAME,
            experiment_name=EXPERIMENT_NAME,
            save_checkpoints=True,
            checkpoint_dir=checkpoint_dir,
            output_file=output_file,
            output_dir=results_dir,
            n_jobs=1,
            validation=True,
        )

        ticker_elapsed = time.time() - ticker_start

        if not results.empty:
            print(f"\n📊 Resultados para {ticker}: {len(results):,} filas en {ticker_elapsed / 60:.1f} min")
            print(f"Mejor Sharpe: {results['sharpe_ratio'].max():.3f}")
            print(f"Peor Sharpe: {results['sharpe_ratio'].min():.3f}")
            print(f"Promedio Sharpe: {results['sharpe_ratio'].mean():.3f}")

            print(f"\n🏆 TOP 10 ESTRATEGIAS ({ticker}):")
            top10 = results.nlargest(10, 'sharpe_ratio')
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                print(f"{idx}. {row['strategy_name']} - Sharpe: {row['sharpe_ratio']:.3f} "
                      f"| Win Rate: {row['win_rate']:.1%} | Total Return: {row['total_return']:.2%}")
        else:
            print(f"⚠️  No se obtuvieron resultados válidos para {ticker}")

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"✅ Proceso completado para todos los activos en {overall_elapsed / 60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
