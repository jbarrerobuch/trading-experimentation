# Estrategias de Trading

Esta carpeta contiene las configuraciones de estrategias en formato YAML para el framework de trading.

## 📁 Estructura de Archivos

Cada archivo `.yaml` define una estrategia que puede ser:
- **Individual**: Un solo indicador con grid de parámetros
- **Combo**: Múltiples indicadores combinados

## 📋 Formato de Estrategia Individual

```yaml
name: Nombre_De_La_Estrategia
type: individual
indicator: nombre_indicador

params_grid:
  parametro1: [valor1, valor2, valor3]
  parametro2: [valor1, valor2]
  position_type: [both, long, short]
```

### Indicadores Disponibles:
- `rsi` - Relative Strength Index
- `macd` - Moving Average Convergence Divergence
- `willr` - Williams %R
- `stoch` - Stochastic Oscillator
- `cci` - Commodity Channel Index
- `rsx` - Relative Strength Xtra
- `cmo` - Chande Momentum Oscillator
- `uo` - Ultimate Oscillator
- `adx` - Average Directional Index
- `er` - Efficiency Ratio
- `slope` - Slope
- `trix` - Triple Exponential Average
- `ao` - Awesome Oscillator
- `inertia` - Inertia
- `bop` - Balance of Power

## 📋 Formato de Estrategia Combo

```yaml
name: Nombre_Combo
type: combo

indicators:
  - indicator: indicador1
    params_grid:
      parametro1: [valor1]
      parametro2: [valor2]
    weight: 1.0
  
  - indicator: indicador2
    params_grid:
      parametro1: [valor1]
    weight: 1.5

combination_methods: [AND, OR, MAJORITY, WEIGHTED]
position_types: [both, long, short]
```

### Métodos de Combinación:
- `AND` - Todos los indicadores deben coincidir
- `OR` - Al menos un indicador da señal
- `MAJORITY` - Mayoría simple (>50%)
- `WEIGHTED` - Promedio ponderado por pesos
- `UNANIMOUS_LONG` - Todos =1 para long, resto 0
- `UNANIMOUS_SHORT` - Todos =-1 para short, resto 0

## 🚀 Uso

### Cargar todas las estrategias:
```python
from trading_strategy import load_all_strategies, strategy_grid_search

configs = load_all_strategies()
results = strategy_grid_search(df, configs)
```

### Cargar estrategias específicas:
```python
from trading_strategy import load_strategies_by_name

configs = load_strategies_by_name(['rsi_optimization', 'macd_optimization'])
results = strategy_grid_search(df, configs)
```

### Ver estrategias disponibles:
```python
from trading_strategy import get_strategy_info

info = get_strategy_info()
print(f"Total: {info['total']}")
print(f"Individuales: {len(info['individual'])}")
print(f"Combos: {len(info['combo'])}")
```

## 📝 Ejemplos Incluidos

- `rsi_optimization.yaml` - Optimización de RSI con múltiples parámetros
- `macd_optimization.yaml` - Optimización de MACD
- `williams_r.yaml` - Williams %R con diferentes períodos
- `rsi_macd_combo.yaml` - Combinación RSI + MACD

## ✍️ Crear Nueva Estrategia

1. Crea un archivo `.yaml` en esta carpeta
2. Define la configuración siguiendo los formatos de arriba
3. Usa `load_all_strategies()` o `load_strategies_by_name(['tu_estrategia'])`
4. Ejecuta grid search normalmente

## 🔍 Validación

El sistema valida automáticamente:
- Campos requeridos (`name`, `type`, `indicator`/`indicators`, `params_grid`)
- Tipos de estrategia válidos
- Estructura de parámetros

Si hay errores, se mostrarán al cargar y la estrategia se saltará.
