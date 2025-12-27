# Quick Start Guide - Trading Strategy Framework

## 🚀 Inicio Rápido (5 minutos)

### 1. Verificar Instalación

```bash
cd d:/py_projects/GammaNeutral/main
python -c "import sys; sys.path.append('src'); from trading_strategy import load_saved_data; print('✅ Módulos OK')"
```

### 2. Ejecutar Ejemplo Básico

```bash
cd examples
python basic_usage.py
```

Esto ejecutará:
- Carga de datos
- Cálculo de indicadores
- Backtest individual
- Grid search (optimización)
- Visualización de resultados
- Exportación a JSON

### 3. Ver Resultados en MLflow

```bash
cd d:/py_projects/GammaNeutral/main
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre: http://localhost:5000

## 📝 Usar en tu Propio Notebook/Script

```python
import sys
sys.path.append('d:/py_projects/GammaNeutral/main/src')

from trading_strategy import (
    load_saved_data,
    calculate_returns_and_momentum,
    strategy_grid_search
)

# 1. Cargar datos
data = load_saved_data('btcusdt', ['1h'])

# 2. Calcular retornos
df = calculate_returns_and_momentum(data['1h'], compute_indicators=False)

# 3. Configurar estrategia
configs = [{
    'name': 'RSI_Simple',
    'indicator': 'rsi',
    'params_grid': {
        'period': [14],
        'overbought': [70],
        'oversold': [30],
        'position_type': ['both']
    }
}]

# 4. Ejecutar grid search
results = strategy_grid_search(
    df, 
    configs, 
    use_mlflow=True,
    ticker='BTCUSDT',
    timeframe='1h'
)

# 5. Ver mejores
print(results.nlargest(5, 'sharpe_ratio'))
```

## 🧪 Ejecutar Tests

```bash
# Instalar pytest si no lo tienes
pip install pytest pytest-cov

# Ejecutar tests
cd d:/py_projects/GammaNeutral/main
pytest tests/ -v

# Con coverage
pytest tests/ --cov=trading_strategy --cov-report=html
# Ver: htmlcov/index.html
```

## 📊 Comandos Útiles

### Ver Estructura del Paquete

```bash
cd d:/py_projects/GammaNeutral/main/src
tree trading_strategy
```

### Importar en Python REPL

```python
>>> import sys
>>> sys.path.append('d:/py_projects/GammaNeutral/main/src')
>>> from trading_strategy import *
>>> help(strategy_grid_search)
```

### Actualizar Módulos sin Reiniciar

```python
import importlib
import trading_strategy
importlib.reload(trading_strategy)
```

## 🐛 Troubleshooting

### Import Error

```python
# Problema: ModuleNotFoundError: No module named 'trading_strategy'

# Solución:
import sys
sys.path.insert(0, 'd:/py_projects/GammaNeutral/main/src')
from trading_strategy import load_saved_data
```

### Pandas-ta Not Found

```bash
pip install pandas-ta
```

### MLflow No Disponible

```bash
pip install mlflow

# O desactivar:
results = strategy_grid_search(df, configs, use_mlflow=False)
```

### CCXT Error

```bash
pip install ccxt --upgrade
```

## 📚 Documentación Adicional

- **README completo**: `src/trading_strategy/README.md`
- **Resumen de modularización**: `MODULARIZATION_SUMMARY.md`
- **Ejemplo completo**: `examples/basic_usage.py`
- **Tests**: `tests/test_basic.py`

## 🎯 Workflows Comunes

### Workflow 1: Exploración Rápida

```python
# En notebook o script
from trading_strategy import *

data = load_saved_data('ethusdt', ['4h'])
df = calculate_returns_and_momentum(data['4h'], compute_indicators=False)

metrics = backtest_strategy(
    df, 
    'rsi', 
    {'period': 14, 'overbought': 70, 'oversold': 30},
    'long'
)

print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
```

### Workflow 2: Optimización Multi-Asset

```python
from trading_strategy import *

tickers = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
all_results = []

for ticker in tickers:
    data = load_saved_data(ticker.lower(), ['1h'])
    df = calculate_returns_and_momentum(data['1h'], compute_indicators=False)
    
    results = strategy_grid_search(df, configs, ticker=ticker, timeframe='1h')
    all_results.append(results)

final = pd.concat(all_results)
print(final.groupby('ticker')['sharpe_ratio'].max())
```

### Workflow 3: Testing de Estrategia Nueva

```python
# 1. Crear función en indicators.py
# 2. Agregar test en tests/test_basic.py
# 3. Ejecutar test
pytest tests/test_basic.py::test_my_new_strategy -v

# 4. Si pasa, usar en grid search
configs = [{
    'name': 'MyNewStrategy',
    'indicator': 'my_new_indicator',
    'params_grid': {...}
}]
```

## 🔗 Links Útiles

- **Pandas-ta Docs**: https://github.com/twopirllc/pandas-ta
- **CCXT Docs**: https://docs.ccxt.com/
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Pytest Docs**: https://docs.pytest.org/

## ✅ Checklist de Inicio

- [ ] Python 3.12+ instalado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Path configurado correctamente
- [ ] Datos descargados (o usar `fetch_ohlcv_data()`)
- [ ] Ejemplo básico ejecutado exitosamente
- [ ] MLflow UI funcionando
- [ ] Tests pasando

## 💡 Tips

1. **Usa `compute_indicators=False`** en grid search para mayor velocidad
2. **Filtra en MLflow** por `tags.ticker` y `tags.timeframe`
3. **Exporta mejores estrategias** a JSON para referencia
4. **Ejecuta tests** antes de hacer cambios grandes
5. **Documenta** tus configuraciones de estrategia en comentarios

---

**¿Problemas?** Revisa `MODULARIZATION_SUMMARY.md` o abre el notebook `momentumEDA.ipynb`
