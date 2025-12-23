"""
Módulo de carga de datos
Maneja descarga y carga de datos OHLCV desde exchanges
"""

import os
import glob
import ccxt
import pandas as pd
from datetime import datetime
from pathlib import Path
from .constants import (
    COL_TIMESTAMP, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME
)
from .utils.paths import get_market_data_dir


def fetch_ohlcv_data(ticker, timeframes=['1h', '4h', '1d'], start_date='2017-08-17', save_data=True):
    """
    Descarga datos OHLCV desde Binance usando CCXT con paginación
    
    Parameters:
    -----------
    ticker : str
        Símbolo del par en formato 'BTC/USDT', 'ETH/USDT', etc.
    timeframes : list
        Lista de timeframes a descargar ['1h', '4h', '1d']
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD'
    save_data : bool
        Si True, guarda los datos en formato Parquet en data/market
    
    Returns:
    --------
    dict : Diccionario con DataFrames por timeframe
    """
    print(f"\n🌐 Descargando datos de {ticker} desde {start_date}...")
    
    # Inicializar exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    # Convertir start_date a timestamp
    start_timestamp = exchange.parse8601(f'{start_date}T00:00:00Z')
    
    # Mapeo de timeframes
    tf_map = {
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '15m': '15m',
        '30m': '30m'
    }
    
    data = {}
    market_dir = get_market_data_dir()
    os.makedirs(market_dir, exist_ok=True)
    
    ticker_clean = ticker.replace('/', '').replace('-', '').lower()
    
    for tf_name in timeframes:
        tf_ccxt = tf_map.get(tf_name, tf_name)
        print(f"\n⏳ Descargando {ticker} @ {tf_name}...", end='')
        
        try:
            # Descargar todos los datos desde start_date
            all_ohlcv = []
            current_timestamp = start_timestamp
            
            while True:
                ohlcv = exchange.fetch_ohlcv(
                    ticker, 
                    timeframe=tf_ccxt, 
                    since=current_timestamp,
                    limit=1000  # Máximo por request
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Actualizar timestamp para siguiente batch
                current_timestamp = ohlcv[-1][0] + 1
                
                # Si llegamos al presente, terminar
                if ohlcv[-1][0] >= exchange.milliseconds():
                    break
            
            # Convertir a DataFrame
            df = pd.DataFrame(
                all_ohlcv,
                columns=[COL_TIMESTAMP, COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]
            )
            
            # Convertir timestamp a datetime
            df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], unit='ms')
            df.set_index(COL_TIMESTAMP, inplace=True)
            
            # Asegurar que los datos sean numéricos
            for col in [COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOLUME]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eliminar duplicados y valores nulos
            df = df[~df.index.duplicated(keep='first')]
            df = df.dropna()
            
            data[tf_name] = df
            print(f" ✓ {len(df)} velas")
            
            # Guardar en formato Parquet
            if save_data:
                date_start = df.index.min().strftime('%Y%m%d')
                date_end = df.index.max().strftime('%Y%m%d')
                parquet_filename = f'{ticker_clean}_{tf_name}_{date_start}_{date_end}.parquet'
                parquet_filepath = market_dir / parquet_filename
                
                df.to_parquet(parquet_filepath, compression='snappy', index=True)
                print(f"    💾 Guardado: {parquet_filepath}")
            
        except Exception as e:
            print(f" ✗ Error: {str(e)}")
            continue
    
    if data:
        print(f"\n✓ Datos descargados exitosamente")
        print(f"  Rango temporal: {min(df.index.min() for df in data.values())} a {max(df.index.max() for df in data.values())}")
        if save_data:
            print(f"  📁 Archivos guardados en: {market_dir}/")
    else:
        print("\n✗ No se pudieron descargar datos")
    
    return data


def load_saved_data(ticker, timeframes=['1h', '4h', '1d'], data_path=None):
    """
    Carga datos guardados en formato Parquet desde el directorio especificado
    
    Parameters:
    -----------
    ticker : str
        Símbolo del par en formato 'BTC/USDT', 'ETH/USDT', etc. o 'btcusdt'
    timeframes : list
        Lista de timeframes a cargar ['1h', '4h', '1d']
    data_path : str or Path, optional
        Ruta al directorio donde se encuentran los archivos Parquet
        Default: None (usa get_market_data_dir())
    
    Returns:
    --------
    dict : Diccionario con DataFrames por timeframe
    """
    print(f"📂 Cargando datos guardados de {ticker}...")
    
    # Nombre limpio del ticker
    ticker_clean = ticker.replace('/', '').replace('-', '').lower()
    
    if data_path is None:
        market_dir = get_market_data_dir()
    else:
        market_dir = Path(data_path)
    
    if not market_dir.exists():
        print(f"❌ Directorio {market_dir} no existe")
        return None
    
    data = {}
    
    for tf in timeframes:
        # Buscar el archivo más reciente para este ticker y timeframe
        pattern = str(market_dir / f'{ticker_clean}_{tf}_*.parquet')
        files = glob.glob(pattern)
        
        if not files:
            print(f"  ⚠️  No se encontraron datos para {tf}")
            continue
        
        # Ordenar por fecha de fin (más reciente primero)
        latest_file = sorted(files)[-1]
        
        try:
            df = pd.read_parquet(latest_file)
            
            # Ensure columns are normalized (in case loading old data)
            # Map lowercase to Capitalized
            rename_map = {
                'open': COL_OPEN, 'high': COL_HIGH, 'low': COL_LOW, 
                'close': COL_CLOSE, 'volume': COL_VOLUME
            }
            df.rename(columns=rename_map, inplace=True)
            
            data[tf] = df
            print(f"  ✓ {tf}: {len(df)} velas ({df.index.min()} a {df.index.max()})")
        except Exception as e:
            print(f"  ✗ Error cargando {tf}: {str(e)}")
    
    if data:
        print(f"✓ Datos cargados exitosamente desde {market_dir}/")
    else:
        print("❌ No se pudieron cargar datos")
    
    return data
