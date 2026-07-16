# Estrategias de Trading

Esta carpeta contiene las configuraciones de estrategias en formato YAML para el framework de trading.

## 📁 Estructura de Archivos

Cada archivo `.yaml` define una estrategia que puede ser:
- **Individual** (`type: individual`): Un solo indicador con grid de parámetros
- **Combo** (`type: combo`): Múltiples indicadores combinados

Organización de carpetas:
- `individual/` — set canónico de estrategias de un solo indicador (una por indicador soportado).
- `combo/`, `selection/` — estrategias combinadas.
- `test/` — configuraciones de prueba.

> **Nota:** el campo `type` solo admite `individual` o `combo`. Cualquier otro
> valor (p. ej. el antiguo `single`) ejecuta el grid search pero se clasifica
> mal en `get_strategy_info()`. Usa siempre `individual` para un solo indicador.

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

### Indicadores Disponibles (28):

Momentum / osciladores:
- `rsi` - Relative Strength Index → `period`, `overbought`, `oversold`
- `rsx` - Relative Strength Xtra → `period`, `overbought`, `oversold`
- `cci` - Commodity Channel Index → `period`, `threshold`
- `cmo` - Chande Momentum Oscillator → `period`
- `willr` - Williams %R → `period`, `high_threshold`, `low_threshold`
- `stoch` - Stochastic Oscillator → `k`, `d`, `smooth_k`
- `stoch_rsi` - Stochastic RSI → `rsi_period`, `k_period`, `d_period`, `overbought`, `oversold`
- `uo` - Ultimate Oscillator → `threshold`
- `ao` - Awesome Oscillator → `fast`, `slow`
- `bop` - Balance of Power → (sin parámetros; solo `position_type`)
- `mfi` - Money Flow Index → `period`, `overbought`, `oversold`
- `mom` - Momentum → `period`
- `roc` - Rate of Change → `period`
- `bbp` - Bollinger Bands %B → `period`, `std_dev`, `lower_threshold`, `upper_threshold`
- `fisher` - Fisher Transform → `period`, `signal` (cruce si `signal` > 1)

Tendencia:
- `macd` - Moving Average Convergence Divergence → `fast`, `slow`, `signal`
- `adx` - Average Directional Index → `period`, `threshold`
- `aroon` - Aroon Oscillator → `period`
- `trix` - Triple Exponential Average → `period`, `signal`
- `slope` - Slope → `period`
- `er` - Efficiency Ratio → `period`, `threshold`
- `inertia` - Inertia → `period`, `upper`, `lower`
- `vortex` - Vortex Indicator → `period`
- `psar` - Parabolic SAR → `af0`, `af`, `max_af`
- `supertrend` - Supertrend → `period`, `multiplier`

Volatilidad / canales:
- `keltner` - Keltner Channels → `period`, `atr_period`, `multiplier`
- `donchian` - Donchian Channels → `period`
- `bbands_width` - Bollinger Bands Width/Expansion → `period`, `std_dev`

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
