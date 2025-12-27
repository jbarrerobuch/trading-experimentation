# Trading Strategy Framework

Framework modular para desarrollo y optimización de estrategias de trading basadas en momentum.

## 📁 Estructura

```
trading_strategy/
├── __init__.py              # Exports principales
├── data_loader.py           # Carga de datos OHLCV (CCXT/Parquet)
├── indicators.py            # Cálculo de 36 indicadores técnicos
├── backtesting.py           # Motor de backtesting con 16 métricas
├── grid_search.py           # Optimización de hiperparámetros
├── visualization.py         # Gráficos y exportación de resultados
└── utils/
    ├── __init__.py
    └── helpers.py           # Funciones auxiliares
```

## 🚀 Instalación

El módulo ya está en el directorio `src/`, simplemente impórtalo:

```python
import sys
sys.path.append('d:/py_projects/GammaNeutral/main/src')

from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    backtest_strategy,
    strategy_grid_search,
    visualize_grid_search_results
)
```

## 📊 Uso Rápido

### 1. Cargar Datos

```python
# Cargar datos guardados
data = load_saved_data('btcusdt', ['1h', '4h', '1d'])

# O descargar nuevos datos
from trading_strategy import fetch_ohlcv_data
data = fetch_ohlcv_data('BTC/USDT', ['1h'], start_date='2020-01-01')
```

### 2. Calcular Indicadores

```python
# Calcular retornos e indicadores
df = calculate_returns_and_momentum(
    data['1h'], 
    compute_indicators=True  # False para solo retornos (más rápido)
)
```

### 3. Backtest Individual

```python
# Configurar estrategia
params = {
    'period': 14,
    'overbought': 70,
    'oversold': 30
}

# Ejecutar backtest
results = backtest_strategy(
    df=df,
    indicator='rsi',
    params=params,
    position_type='both'
)

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Win Rate: {results['win_rate']:.1%}")
```

### 4. Grid Search (Optimización)

```python
# Configurar grid search
configs = [
    {
        'name': 'RSI_Optimization',
        'indicator': 'rsi',
        'params_grid': {
            'period': [10, 14, 20],
            'overbought': [70, 75, 80],
            'oversold': [20, 25, 30],
            'position_type': ['long', 'short', 'both']
        }
    }
]

# Ejecutar optimización
results = strategy_grid_search(
    df=df,
    strategy_configs=configs,
    use_mlflow=True,
    ticker='BTCUSDT',
    timeframe='1h'
)

# Ver mejores estrategias
print(results.nlargest(5, 'sharpe_ratio'))
```

### 5. Visualizar Resultados

```python
from trading_strategy import visualize_grid_search_results, export_best_strategies

# Gráficos
visualize_grid_search_results(results)

# Exportar mejores
export_best_strategies(results, top_n=10)
```

## 🎯 Indicadores Soportados

### Osciladores (8)
- `rsi`: Relative Strength Index
- `roc`: Rate of Change
- `mom`: Momentum
- `macd`: MACD
- `stoch`: Stochastic
- `cci`: Commodity Channel Index
- `willr`: Williams %R
- `cmo`: Chande Momentum Oscillator

### Tendencia (6)
- `adx`: Average Directional Index
- `slope`: Slope (Linear Regression)
- `trix`: Triple Exponential Average
- `ao`: Awesome Oscillator
- `er`: Efficiency Ratio
- `bop`: Balance of Power

### Avanzados (3)
- `rsx`: RSI Smoothed
- `inertia`: Inertia Indicator
- `uo`: Ultimate Oscillator

## 🔄 Estrategias Multi-Indicador

```python
# Configurar combo
combo_config = {
    'name': 'RSI_MACD_Combo',
    'type': 'combo',
    'indicators': [
        {
            'indicator': 'rsi',
            'params_grid': {
                'period': [14],
                'overbought': [70],
                'oversold': [30]
            }
        },
        {
            'indicator': 'macd',
            'params_grid': {
                'fast': [12],
                'slow': [26],
                'signal': [9]
            }
        }
    ],
    'combination_methods': ['AND', 'OR', 'MAJORITY'],
    'position_types': ['both']
}

# Ejecutar grid search con combo
results = strategy_grid_search(df, [combo_config], ticker='BTCUSDT', timeframe='1h')
```

### Métodos de Combinación

- **AND**: Todos los indicadores deben estar de acuerdo
- **OR**: Al menos un indicador da señal
- **MAJORITY**: Mayoría simple (>50%)
- **WEIGHTED**: Promedio ponderado por pesos
- **UNANIMOUS_LONG**: Todos =1 para long
- **UNANIMOUS_SHORT**: Todos =-1 para short

## 📈 Métricas Calculadas (16)

1. **total_return**: Retorno total acumulado
2. **n_trades**: Número de operaciones
3. **hit_rate**: % de trades positivos
4. **win_rate**: % de trades ganadores
5. **sharpe_ratio**: Retorno ajustado por riesgo
6. **sortino_ratio**: Sharpe con downside deviation
7. **calmar_ratio**: Retorno / |Max Drawdown|
8. **max_drawdown**: Peor caída desde peak
9. **profit_factor**: Total gain / Total loss
10. **avg_win**: Promedio de ganancias
11. **avg_loss**: Promedio de pérdidas
12. **risk_reward_ratio**: |Avg Win / Avg Loss|
13. **best_trade**: Mejor trade
14. **worst_trade**: Peor trade
15. **avg_return_per_trade**: Retorno promedio
16. **volatility**: Desviación estándar de retornos

## 🎛️ MLflow Integration

El framework integra MLflow automáticamente para tracking de experimentos:

```python
# Los experimentos se guardan en: mlflow.db (SQLite)
# Nombre del experimento: momentum_trading_strategies

# Para ver en UI:
# mlflow ui --backend-store-uri sqlite:///mlflow.db
# Abre: http://localhost:5000

# Filtrar en MLflow:
# tags.ticker = "BTCUSDT" AND tags.timeframe = "1h"
# metrics.sharpe_ratio > 1.0
```

## 🌍 Multi-Asset / Multi-Timeframe

```python
tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
timeframes = ['1h', '4h', '1d']
all_results = []

for ticker in tickers:
    for tf in timeframes:
        data = load_saved_data(ticker.lower(), [tf])
        data[tf] = calculate_returns_and_momentum(data[tf], compute_indicators=False)
        
        results = strategy_grid_search(
            data[tf], 
            configs, 
            ticker=ticker, 
            timeframe=tf
        )
        all_results.append(results)

# Consolidar
final_df = pd.concat(all_results)

# Análisis por asset
print(final_df.groupby('ticker')['sharpe_ratio'].max())
```

## 🐛 Debug

Los módulos facilitan el debug al separar responsabilidades:

```python
# Debug data loading
from trading_strategy.data_loader import load_saved_data
data = load_saved_data('btcusdt', ['1h'])

# Debug indicators
from trading_strategy.indicators import calculate_indicator_and_signals
signals = calculate_indicator_and_signals(data['1h'], 'rsi', {'period': 14, 'overbought': 70, 'oversold': 30})

# Debug backtest
from trading_strategy.backtesting import backtest_strategy
metrics = backtest_strategy(data['1h'], 'rsi', {'period': 14, 'overbought': 70, 'oversold': 30}, 'long')
```

## 📝 Ejemplo Completo

Ver `examples/basic_usage.py` para un ejemplo completo de uso.

## ⚙️ Configuración

### Dependencias

```
pandas >= 2.0
numpy >= 1.24
pandas-ta >= 0.4.71
ccxt >= 4.0
matplotlib >= 3.7
seaborn >= 0.13
mlflow (opcional)
```

### Variables de Entorno

```python
# Opcional: configurar MLflow tracking URI
import os
os.environ['MLFLOW_TRACKING_URI'] = 'file:../mlruns'
```

## 🤝 Contribuciones

Este es un framework interno. Para mejoras:
1. Editar módulos en `src/trading_strategy/`
2. Actualizar tests si aplica
3. Documentar cambios en este README

## 📄 Licencia

Uso interno - GammaNeutral Project
