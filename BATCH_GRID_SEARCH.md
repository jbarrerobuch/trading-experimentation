# 🔄 Sistema de Batch Grid Search

Optimización de memoria y checkpoints para experimentos grandes (50K-200K+ runs)

---

## 🎯 ¿Por qué usar Batches?

### Problemas sin Batching:
```python
# Ejecutar 200K experimentos en una sola ejecución
results = strategy_grid_search(df, configs, experiment_name='huge_test')
```

**Problemas:**
- ❌ **Memoria**: 200K resultados en RAM (~95MB + overhead MLflow ~50MB)
- ❌ **Sin checkpoints**: Si falla en run 150K → pierdes todo
- ❌ **Sin visibilidad**: No puedes ver resultados hasta que termine
- ❌ **Bloqueo**: VS Code puede crashear con MLflow tracking

### Ventajas con Batching:
```python
# Dividir en 20 batches de 10K
results = batch_grid_search(df, configs, batch_size=10000)
```

**Beneficios:**
- ✅ **Memoria constante**: ~5MB por batch (libera después de cada uno)
- ✅ **Checkpoints automáticos**: Recuperas desde el último batch completado
- ✅ **Visibilidad**: Ves resultados intermedios en MLflow UI
- ✅ **Estabilidad**: No sobrecarga memoria de VS Code

---

## 📊 Comparativa de Recursos

### Experimento: 100K runs

| Método           | Memoria Peak| Tiempo | Checkpoints    | Recuperable |
|--------          |-------------|--------|----------------|-------------|
| **Sin Batching** | ~150 MB     | 25 min | ❌ No          | ❌ No      |
| **Batch 20K**    | ~10 MB      | 26 min | ✅ 5 archivos  | ✅ Sí      |
| **Batch 10K**    | ~5 MB       | 27 min | ✅ 10 archivos | ✅ Sí      |
| **Batch 5K**     | ~3 MB       | 28 min | ✅ 20 archivos | ✅ Sí      |

**Recomendación**: Batch size 10K-20K (equilibrio memoria/overhead)

---

## 🚀 Guía de Uso

### 1. Uso Básico

```python
from src.trading_strategy import (
    load_saved_data,
    load_all_strategies,
    batch_grid_search
)

# Cargar datos
df = load_saved_data('BTCUSDT', '1h')

# Cargar estrategias
configs = load_all_strategies()

# Ejecutar con batches
results = batch_grid_search(
    df=df,
    strategy_configs=configs,
    batch_size=10000,              # 10K por batch
    experiment_name='btc_optimization',
    save_checkpoints=True          # Guardar checkpoints
)
```

### 2. Estimar Recursos Antes de Ejecutar

```python
from src.trading_strategy import estimate_batch_requirements

# Ver cuántos batches se generarán
estimate_batch_requirements(configs, batch_size=10000)
```

**Salida:**
```
📊 ESTIMACIÓN DE RECURSOS
================================================================================
Batch size propuesto: 10,000 experiments

rsi_optimization:
  Experiments: 48,000
  Batches: 5

macd_optimization:
  Experiments: 20,000
  Batches: 2

────────────────────────────────────────────────────────────────────────────────
📈 TOTALES:
  Total experiments: 68,000
  Total batches: 7
  Estimated time: 17.0 min (0.3 hrs)
  Peak memory per batch: ~4.9 MB
  Total results size: ~33.2 MB
```

### 3. Recuperación desde Checkpoint

Si tu ejecución falla o se interrumpe, puedes reanudar:

```python
import pandas as pd

# Cargar checkpoints guardados
batch1 = pd.read_csv('checkpoint_btc_optimization_20251123_143052_batch1.csv')
batch2 = pd.read_csv('checkpoint_btc_optimization_20251123_143052_batch2.csv')
# ... cargar todos los checkpoints disponibles

# Combinar
results = pd.concat([batch1, batch2, ...], ignore_index=True)

# Continuar desde donde quedó (ejecutar solo batches faltantes)
# → Requiere identificar qué configs ya se ejecutaron
```

### 4. Ajustar Batch Size Según Recursos

```python
# Para sistemas con poca RAM (< 8GB)
results = batch_grid_search(df, configs, batch_size=5000)

# Para sistemas normales (8-16GB)
results = batch_grid_search(df, configs, batch_size=10000)

# Para sistemas potentes (32GB+)
results = batch_grid_search(df, configs, batch_size=20000)
```

---

## 🔍 Análisis de Resultados por Batch

### Ver Resultados Acumulados en MLflow

```python
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
experiment = mlflow.get_experiment_by_name('btc_optimization')

# Ver todos los runs (de todos los batches)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.sharpe_ratio DESC"]
)

# Filtrar por sesión específica
session_runs = runs[runs['tags.session_id'] == '20251123_143052']
print(f"Runs en esta sesión: {len(session_runs)}")

# Ver evolución por batch
by_batch = session_runs.groupby('tags.batch_id').agg({
    'metrics.sharpe_ratio': ['count', 'mean', 'max']
})
print(by_batch)
```

### Comparar Sesiones Diferentes

```python
# Todas las sesiones en el mismo experimento
sessions = runs['tags.session_id'].unique()
print(f"Sesiones registradas: {len(sessions)}")

for session_id in sessions:
    session_data = runs[runs['tags.session_id'] == session_id]
    best_sharpe = session_data['metrics.sharpe_ratio'].max()
    print(f"Sesión {session_id}: {len(session_data)} runs, mejor Sharpe: {best_sharpe:.3f}")
```

---

## 💡 Best Practices

### 1. Nombra tus experimentos consistentemente
```python
# Incluye fecha o versión
experiment_name='btc_1h_v2_20251123'

# Incluye parámetros importantes
experiment_name='btc_1h_optimized_cache_enabled'
```

### 2. Usa batch_size apropiado
```python
# Muy pequeño → mucho overhead de I/O
batch_size=1000  # ❌ 100 batches para 100K runs

# Muy grande → mucha memoria
batch_size=50000  # ❌ No libera memoria frecuentemente

# Óptimo
batch_size=10000  # ✅ Balance perfecto
```

### 3. Monitorea durante ejecución
```python
# Abre MLflow UI en otra terminal mientras corre
mlflow ui --backend-store-uri file:./mlruns

# Abre http://localhost:5000 en navegador
# → Verás runs agregándose en tiempo real
```

### 4. Guarda checkpoints en producciones largas
```python
# Siempre activar para experimentos >50K
results = batch_grid_search(
    df, configs,
    batch_size=10000,
    save_checkpoints=True  # ✅ Siempre True para experimentos grandes
)
```

---

## 🛠️ Troubleshooting

### Problema: "Out of Memory" durante batch

**Solución**: Reducir batch_size
```python
# De 20K a 10K
batch_size=10000
```

### Problema: Batches muy lentos

**Causa**: I/O overhead de muchos batches pequeños

**Solución**: Aumentar batch_size
```python
# De 5K a 15K
batch_size=15000
```

### Problema: Checkpoints no se guardan

**Causa**: Permisos de escritura o path inválido

**Solución**: Verificar permisos
```python
import os
print(os.access('.', os.W_OK))  # Debe ser True
```

### Problema: Sesión duplicada en MLflow

**Causa**: Múltiples ejecuciones simultáneas con mismo experiment_name

**Solución**: Usar experiment_name único
```python
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name=f'btc_optimization_{timestamp}'
```

---

## 📈 Ejemplo Completo: 200K Experimentos

```python
from src.trading_strategy import (
    load_saved_data,
    load_all_strategies,
    batch_grid_search,
    estimate_batch_requirements
)

# 1. Cargar datos
df = load_saved_data('BTCUSDT', '1h', start_date='2017-08-18', end_date='2025-11-11')

# 2. Cargar estrategias (genera ~200K combinaciones)
configs = load_all_strategies()

# 3. Estimar recursos
print("Estimando recursos para 200K experimentos...")
estimate_batch_requirements(configs, batch_size=10000)
# Output: ~20 batches, ~50 min, ~5MB peak memory

# 4. Ejecutar
print("\nIniciando batch grid search...")
results = batch_grid_search(
    df=df,
    strategy_configs=configs,
    batch_size=10000,
    experiment_name='btc_200k_full_optimization',
    save_checkpoints=True
)

# 5. Analizar top 50
top50 = results.nlargest(50, 'sharpe_ratio')
top50.to_csv('top50_strategies.csv', index=False)

print(f"\n✅ Completado: {len(results):,} experimentos")
print(f"🏆 Mejor Sharpe: {results['sharpe_ratio'].max():.3f}")
```

**Resultado esperado:**
- ✅ 20 batches ejecutados
- ✅ 20 checkpoints guardados
- ✅ ~5MB memoria constante
- ✅ ~50 minutos tiempo total
- ✅ 200K runs en MLflow
- ✅ Sin crashes

---

## 🔗 Integración con Sistema Existente

El sistema de batches es **completamente compatible** con el workflow normal:

```python
# Forma anterior (sin batches) - sigue funcionando
from src.trading_strategy import strategy_grid_search

results = strategy_grid_search(df, configs, experiment_name='test')


# Forma nueva (con batches) - misma interfaz, mejor rendimiento
from src.trading_strategy import batch_grid_search

results = batch_grid_search(df, configs, batch_size=10000, 
                           experiment_name='test')


# Ambas devuelven el mismo DataFrame
# Ambas registran en MLflow
# Ambas usan las mismas estrategias YAML
```

---

## 📚 Archivos Relacionados

- **`src/trading_strategy/batch_grid_search.py`**: Implementación del sistema
- **`examples/batch_usage.py`**: Ejemplo completo de uso
- **`examples/basic_usage.py`**: Ejemplo sin batches (para comparar)

---

## 🎓 Recomendaciones Finales

### Cuándo usar batches:
- ✅ Experimentos > 50K runs
- ✅ RAM limitada (< 16GB)
- ✅ Necesitas checkpoints
- ✅ Quieres ver progreso en MLflow UI

### Cuándo NO usar batches:
- ❌ Experimentos < 10K runs
- ❌ Prototipado rápido
- ❌ Debugging de estrategias

**Regla simple**: Si tienes duda, **usa batches**. El overhead es mínimo (~2-5%) y ganas estabilidad.
