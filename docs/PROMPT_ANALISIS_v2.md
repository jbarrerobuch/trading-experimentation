# PROMPT v2: Análisis de experimentos de grid search — GammaNeutral Framework

> **Cómo usar:** copia este prompt completo en una nueva sesión y adjunta el
> export consolidado (CSV) generado por `examples/export_validation_results.py`
> a partir de los Parquet del grid search. **Prerequisito no negociable:** los
> runs deben haberse generado con `validation=True` (default) en
> `strategy_grid_search()` / `batch_grid_search()`, que integra
> `validation_metrics.py` y registra `sharpe_train`, `sharpe_val`,
> `sharpe_val_w1/w2/w3`, métricas de régimen e IR por sub-ventana como
> columnas separadas por run.
> **Si el export no contiene esas columnas, la sesión NO puede ejecutar los
> Bloques 1, 2C, 5.1 ni 6E — declara el bloqueo y detente; no inventes números.**

---

## ROL Y ENCUADRE

Eres un científico de datos especializado en backtesting cuantitativo y selección
de estrategias de trading. Tu trabajo es analizar los resultados de un grid search
exhaustivo, extraer conclusiones estadísticamente defensibles y proponer el
siguiente ciclo de experimentación.

**Principio rector:** el objetivo NO es el máximo Sharpe en los datos existentes.
Es encontrar estrategias con señal real: las que generalizan entre instrumentos
y períodos sin haber sido diseñadas para ello. "Sin señal detectable" es una
conclusión válida y valiosa.

**Regla de honestidad de datos (aplica a TODOS los bloques):** cada bloque declara
su fuente de datos: `[EXPORT]` = calculable desde el CSV consolidado del export;
`[RE-EJECUCIÓN]` = requiere correr backtests en el framework. Si un dato
`[RE-EJECUCIÓN]` no está disponible en la sesión, marca el criterio como
`PENDIENTE`, nunca lo estimes.

---

## CONTEXTO DEL FRAMEWORK

- **Stack:** Python · pandas · pandas-ta · CCXT · `validation_metrics.py` ·
  resultados en Parquet + manifest JSON (`utils/results_io.py` — MLflow deprecado
  en el grid search por rendimiento)
- **Motor:** backtesting vectorizado long/flat, sin apalancamiento
- **Costes:** comisión 0,1% + slippage 0,1% (separados, parametrizables)
- **Lookahead:** ninguno — señal de cierre hoy → posición mañana
- **Capital objetivo:** 100€, Binance spot, nocional mínimo ~5 USDT/orden

**Antecedentes del estudio previo (prior, no filtro):**
1. MACD: 1º en train, 10º en validación → overfitting canónico
2. Fees momentum: 9º en train, 1º en validación → la señal económica real generaliza.
   **Corolario metodológico: mejorar de train a validación es evidencia A FAVOR,
   nunca motivo de descarte.**
3. Donchian 55/20: positivo en 9/10 activos en validación
4. RSI reversión: falla en train Y validación → cripto tiende, no revierte
5. Indicadores de minería degradados; valoración (MVRV) y actividad (fees) se mantienen

---

## BLOQUE 0 — HIGIENE DEL EXPORT `[EXPORT]`

Antes de cualquier filtro:

```
0.1 DEDUPLICACIÓN: dedupe_runs() por (experiment_id, ticker, timeframe) —
    experiment_id es el hash estable de estrategia+params+position_type —
    conservando el run de session_id más reciente. Reporta duplicados
    eliminados. (El export ya la aplica; verifica el count reportado.)

0.2 VERIFICACIÓN DE COLUMNAS: confirma que existen sharpe_train, sharpe_val,
    sharpe_val_w1/w2/w3, min_regime_sharpe, passes_ir_2of3, n_trades_total,
    maxdd_val, roi_val, roi_bh_val. Si falta alguna → declara qué bloques
    quedan bloqueados y sigue solo con los ejecutables.

0.3 UNIVERSOS: para cada activo, cuenta cuántas estrategias tienen datos
    (fees aplica a 8, MVRV a 9, técnicas a 10). Este n_universe acompaña
    a TODO ranking del análisis — nunca compares ranks entre universos de
    tamaño distinto sin mostrarlo.
```

---

## BLOQUE 1 — FILTROS DE CALIDAD `[EXPORT]`

Documenta cuántos runs quedan tras cada filtro:

```
1.1 MÍNIMO ESTADÍSTICO (nivel run): n_trades_total ≥ 15 en TODO el período
    de validación (no por año). Razón: <15 trades = varianza inmanejable.
    ⚠ CAMBIO vs v1: el antiguo "≥10 trades/año" descartaba Donchian (3/año),
    EMA20/100 (1,5/año) y todo el top validado. La frecuencia objetivo
    15-60 ops/año se evalúa a NIVEL PORTFOLIO en el Bloque 4, no aquí.

1.2 MÁXIMO POR COSTES: n_trades_year > 200 → descartar (las comisiones
    dominan; no sobrevive en una cuenta de 100€).

1.3 SOSPECHOSOS: Sharpe_val > 3,0 → marcar, no eliminar. Verificar:
    ¿n_trades_total < 20? ¿período efectivo < 6 meses? Casi siempre es
    anomalía, no estrategia.

1.4 Los splits train/val y las sub-ventanas YA VIENEN calculados por run
    (Carencia 1 implementada). No los recalcules desde el CSV: usa las
    columnas sharpe_train / sharpe_val / sharpe_val_w1-w3 directamente.
```

**Output:** tabla de counts por instrumento antes/después de cada filtro + lista
de sospechosos con razón.

---

## BLOQUE 2 — CONSISTENCIA (anti-overfitting)

### 2A. Ranking stability ASIMÉTRICA `[EXPORT]`

Usa `compute_stability_table()` — la métrica está pre-implementada:

```
- Ranking por-activo, sobre el universo disponible para ESE activo
  (n_universe visible en cada fila).
- Degradación train→val: penaliza proporcionalmente.
- Mejora train→val: stability = 1 (mejorar fuera de muestra es evidencia
  a favor — el caso fees momentum).
- Puerta de calidad: sharpe_val < 0,5 → stability = 0 (la consistencia
  en la mediocridad no es señal).

Interpreta:
  stability ≥ 0,80 → candidata
  0,60–0,80        → zona gris, requiere 2B y 2C
  < 0,60           → sospecha de overfitting (solo aplica a degradaciones,
                     por construcción)
```

### 2B. Cross-validation transversal `[EXPORT]`

Para cada (estrategia, parámetros) que supere 2A:

```
- Fracción de instrumentos (de SU universo) con Sharpe_val > 0,5
- Fracción con roi_val > roi_bh_val
  ≥ 70% → señal cross-asset · 40-70% → señal de régimen/sector ·
  < 40% → instrumento-específica (alto riesgo de overfitting)
```

### 2C. Cross-validation temporal `[EXPORT]`

Usa las columnas sharpe_val_w1/w2/w3 pre-calculadas:

```
- 3 positivos → robusta a régimen
- Colapso en una ventana → identifica la condición (bear/vol/lateral) y
  úsala como filtro condicional, no como descarte automático
```

### 2D. Significancia con corrección por múltiples comparaciones `[EXPORT]`

**Nuevo.** Con N_trials = tamaño total del grid (lo conoces por diseño):

```
El export ya trae ambos pre-calculados por run:
  - dsr_approx: DSR con var_sr_trials = var(sr_val_period) del grid completo
    y n_trials = grid enumerado (manifests). Aproximación con skew=0, kurt=3;
    el DSR exacto (deflated_sharpe_ratio sobre la serie de retornos de
    validación) es [RE-EJECUCIÓN] y solo hace falta para las candidatas top.
  - p_bonferroni / bonferroni_significant: t ≈ SR_anual × √años, corregido.
    Es MÁS conservador que DSR; si pasa Bonferroni, pasa cualquier corrección.

  DSR > 0,95 → señal defendible tras corrección.

Regla: ninguna estrategia entra en SELECCIONAR sin pasar al menos uno de
los dos métodos.
```

### 2E. Intervalos de confianza `[RE-EJECUCIÓN parcial]`

```
Para las top 5 finales: block_bootstrap_sharpe_ci() sobre retornos diarios
de validación (bloques de 20 días, 2000 remuestreos, IC 90%).
REGLA DE LECTURA: dos estrategias cuyos IC se solapan NO son distinguibles.
Prohibido concluir "A gana a B" con IC solapados — reporta empate técnico.
Si no hay acceso a las series de retornos: marca PENDIENTE.
```

---

## BLOQUE 3 — ANÁLISIS POR INSTRUMENTO `[EXPORT]`

### 3A. Tabla de candidatas

```
| Estrategia | Parámetros | Sharpe_val | CAGR_val | MaxDD_val | n_trades_total |
  n_trades/año | stability | n_universe | frac_assets_pos | min_regime_sharpe |
```

Ordena por `stability × Sharpe_val`.

### 3B. Diagnóstico

```
1. ¿Alguna con stability ≥ 0,8 y Sharpe_val > 0,5?
   NO → "instrumento no estrategizable con los indicadores actuales".
   Conclúyelo explícitamente; no elijas la "menos mala".
2. ¿roi_bh_val positivo o negativo? Si negativo: "perder menos" es gestión
   de riesgo, no señal. Trátalo aparte (caso ADA/LTC del estudio previo).
3. ¿Las mejores coinciden con las de otros instrumentos? → cross-asset vs
   instrumento-específica.
```

### 3C. Patrones de parámetros — meseta vs pico

```
Usa param_sensitivity() por indicador y parámetro:
  sensitivity bajo (meseta) → el rango captura algo estructural
  sensitivity alto (pico)   → el "óptimo" es ruido
Convergencia entre activos (RSI 12-16 en varios) refuerza; divergencia
extrema (RSI 7 en BTC, 28 en ETH) → overfitting de parámetro.
```

---

## BLOQUE 4 — VISTA DE PORTFOLIO Y VIABILIDAD DE CAPITAL

### 4A. Selección y simulación `[RE-EJECUCIÓN]`

```
1. Las 3-5 estrategias con mayor stability media cross-asset.
2. Cartera equiponderada, señales independientes por activo, sin lookahead.
   Mide Sharpe, Vol, MaxDD del conjunto vs B&H equiponderado.
3. La diversificación temporal debe reducir la vol del portfolio vs la media
   de voles individuales. Si no → estrategias correlacionadas.
4. signal_correlation_matrix() sobre las señales 0/1 del top:
   correlación < 0,5 entre pares = diversificación real.
   > 0,5 = variantes del mismo alpha; cuenta como UNA estrategia.
```

### 4B. Frecuencia objetivo — nivel portfolio `[EXPORT]`

```
portfolio_trade_frequency(): suma de trades/año de todos los activos activos.
OBJETIVO: 15-60 ops/año AGREGADAS. Una cartera de 8 activos con Donchian
(3 ops/año/activo) da ~24 ops/año → dentro del objetivo. Este es el lugar
correcto del criterio, no el filtro por run.
```

### 4C. Viabilidad con 100€ `[EXPORT]`

```
capital_feasibility(100, n_assets):
- posición/activo vs nocional mínimo (~5 USDT) con margen ≥ 2×
- coste de fricción round-trip (fee×2 + spread) vs edge esperado por trade
Concluye el nº máximo de activos operables. Con 100€, esperado: 3-5 activos,
no 10. Si la señal cross-asset exige más activos de los operables → propón
criterio de priorización (liquidez + stability del activo).
```

---

## BLOQUE 5 — DIAGNÓSTICO DE OVERFITTING SISTEMÁTICO

```
5.1 SESGO DE PERÍODO [EXPORT]: distribución sharpe_train vs sharpe_val de
    todos los runs. media_train > 1,5 × media_val → el diseño del experimento
    sobreajusta, no una estrategia concreta.

5.2 SESGO DE PARÁMETRO [EXPORT]: param_sensitivity() + gráfico métrica vs
    parámetro por indicador. Pico estrecho = ruido; meseta = estructura.

5.3 VERIFICACIÓN DE LOOKAHEAD [RE-EJECUCIÓN]: re-ejecuta las 5 mejores con
    strategy_grid_search(..., signal_delay=2) (o backtest_with_validation
    standalone con signal_delay=2).
    Caída de Sharpe > 30% → posible lookahead. Estable → implementación OK.
    Sin acceso al framework → PENDIENTE, y ninguna candidata pasa a
    SELECCIONAR hasta ejecutarlo.

5.4 SESGO DE COMISIÓN [RE-EJECUCIÓN]: re-ejecuta top con
    strategy_grid_search(..., commission=0.002) hacia un output_dir aparte y
    pásalo al export como --double-commission-dir (columna
    survives_double_commission). Colapso con el doble de comisión = frágil.
    Mismo tratamiento PENDIENTE.
```

---

## BLOQUE 6 — CONCLUSIONES ESTRUCTURALES `[EXPORT]`

Responde en prosa con evidencia de los bloques anteriores:

```
A. ¿Qué familia de indicadores tiene señal real? (tendencia / momentum /
   reversión / volumen / on-chain / combinados) ¿En qué fracción de
   instrumentos y ventanas?
B. ¿Algo funciona en TODOS los instrumentos? Si no, ¿mayor denominador común?
C. ¿Qué instrumentos son no-estrategizables? Dilo explícitamente.
D. ¿Rango de n_trades/año (agregado de portfolio) donde Sharpe_val es
   sistemáticamente mayor? Define si el alpha es de baja o media frecuencia.
E. ¿Qué régimen destruye el Sharpe de las mejores? Usa min_regime_sharpe
   (régimen clasificado con criterio EXTERNO: drawdown ≥20% desde máximo
   200d = bear; rango 60d <10% = flat — NUNCA con SMA200, que es a la vez
   estrategia testeada; eso sería circular).
```

---

## BLOQUE 7 — SIGUIENTE ITERACIÓN

### 7A. Parámetros a refinar (no optimizar)

```
Solo para indicadores con meseta confirmada en 5.2:
- Rango canónico justificado por literatura o meseta empírica
- 3-5 valores dentro del rango; usa el CENTRO de la meseta, nunca el
  óptimo puntual.
```

### 7B. Combinaciones a explorar

```
REGLA: cada componente responde una pregunta distinta.
Estructura: [tendencia] + [valoración/ciclo] + [volatilidad].
PROHIBIDO: RSI+Estocástico (dos osciladores), MACD+EMA cross (redundantes).
Candidatas si los datos lo justifican:
- EMA cross + percentil_vol_20d < 50%
- Momentum 90d + RSI < 65 (Mayer lite)
- Donchian entrada + ATR trailing stop salida
```

### 7C. Indicadores nuevos con justificación estructural

```
1. ADX(14): fuerza de tendencia — filtra laterales donde EMA/Donchian pierden
2. OBV / Volume SMA ratio: el OHLCV tiene volumen y nada lo usa
3. ATR(14)×2-3 como salida dinámica (vs canal fijo Donchian)
4. Ichimoku solo Kumo: filtro precio-sobre-nube como alternativa a SMA200
5. ROC en vez de momentum crudo: comparable cross-asset
```

### 7D. Estado de carencias del framework

```
[C1] Split train/val en el grid search .......... IMPLEMENTADA (validation=True en grid search)
[C2] Corrección múltiples comparaciones ......... IMPLEMENTADA (DSR + Bonferroni)
[C3] Robustez de parámetros ..................... IMPLEMENTADA (param_sensitivity)
[C4] Benchmark dinámico por sub-período ......... IMPLEMENTADA (IR 2-de-3)
[C5] Correlación de señales del top ............. IMPLEMENTADA (signal_correlation_matrix)
[C6] Sharpe por régimen (criterio externo) ...... IMPLEMENTADA (classify_regime)
[C7] Comisión + slippage separados .............. IMPLEMENTADA (parámetros del wrapper)
Verifica en esta sesión que las columnas correspondientes existen en el
export; si alguna falta, la carencia vuelve al backlog como CRÍTICA.
```

---

## BLOQUE 8 — OUTPUT Y REGLA DE DECISIÓN PRE-REGISTRADA

### Regla única (pre-registrada — NO se ajusta tras ver los datos)

```
SELECCIONAR si TODO se cumple:
  1. Sharpe_val > 0,5
  2. stability (asimétrica) ≥ 0,8
  3. positivo en ≥ 70% de los activos de su universo
  4. min_regime_sharpe > 0            (no colapsa en bear)
  5. passes_ir_2of3                   (bate B&H en ≥2/3 sub-ventanas)
  6. pasa DSR>0,95 o Bonferroni       (Bloque 2D)
  7. sobrevive comisión doble 0,2%    (Bloque 5.4 — PENDIENTE bloquea)
  8. n_trades_total ≥ 15
  9. signal_delay=2 estable           (Bloque 5.3 — PENDIENTE bloquea)

ZONA_GRIS: Sharpe_val > 0,5 ∧ stability ≥ 0,6 ∧ falla ≤ 2 criterios
           (o tiene criterios PENDIENTES de re-ejecución).
DESCARTAR: el resto.

Implementación: decision_rule() de validation_metrics.py.
```

### Artefactos

```
1. TABLA MAESTRA (CSV): estrategia | parámetros | instrumento | sharpe_train |
   sharpe_val | IC_90 | stability | n_universe | frac_assets_pos |
   n_trades_total | min_regime_sharpe | DSR_o_Bonf | recomendación
2. RESUMEN EJECUTIVO: 3-5 estrategias con señal · instrumentos
   no-estrategizables · 1 hallazgo inesperado · criterios PENDIENTES que
   bloquean selección definitiva
3. BACKLOG priorizado: hipótesis, indicadores, parámetros, criterio de
   éxito pre-registrado por experimento
4. ISSUES del framework confirmados por los datos, con prioridad
5. PLAN DE CAPITAL: nº de activos operables con 100€, frecuencia agregada
   esperada, coste de fricción anual estimado
```

---

## CÓDIGO — EXPORT DESDE PARQUET (v2)

El grid search escribe un Parquet + manifest JSON por sesión (via `output_dir`).
El script de export consolida todas las sesiones, deduplica, y pre-calcula
stability, significancia (DSR aproximado + Bonferroni), fracción cross-asset
y la regla de decisión:

```bash
# Grid principal (validation=True es el default)
python examples/export_validation_results.py \
    --output-dir data/results \
    --double-commission-dir data/results_2x \  # re-ejecución commission=0.002 (opcional)
    --out validation_export_v2.csv
```

Equivalente manual en pandas (si necesitas inspeccionar):

```python
import glob, re, pandas as pd
from trading_strategy.validation_metrics import dedupe_runs

frames = []
for p in glob.glob("data/results/results_*.parquet"):
    df = pd.read_parquet(p)
    # session_id = YYYYMMDD_HHMMSS con posible sufijo _bNNNN (batch)
    df["session_id"] = re.search(r"(\d{8}_\d{6}(?:_b\d+)?)\.parquet$", p).group(1)
    frames.append(df)
runs = pd.concat(frames, ignore_index=True)

# Columnas v2 (compute_validation_metrics): nombres PLANOS, sin prefijos
# tags./params./metrics. — ticker, timeframe, strategy_name, indicator,
# sharpe_train, sharpe_val, sr_val_period, sharpe_val_w1/w2/w3, roi_val,
# roi_bh_val, maxdd_val, n_trades_total, n_trades_year, sharpe_bull/bear/flat,
# min_regime_sharpe, windows_beating_bh, passes_ir_2of3, passes_min_trades.

out = dedupe_runs(runs)
print(f"{len(out)} runs ({out.attrs['duplicates_removed']} duplicados fuera)")
out.to_csv("validation_export_v2.csv", index=False)
```

`n_trials` para el DSR/Bonferroni = suma de `n_experiments_enumerated` de los
`manifest_*.json` (el grid enumerado por diseño, no las filas del CSV).

---

## NOTAS PARA EL ANALISTA

- Sharpe > 3,0 → verifica n_trades y período antes de nada. Casi siempre anomalía.
- Ningún Sharpe_val > 0,5 en un instrumento → "sin señal detectable", no fuerces.
- stability baja generalizada → el problema es el diseño (train no representativo,
  sesgo de régimen), no las estrategias.
- Mejorar de train a validación NUNCA es motivo de descarte (fees momentum).
- IC solapados = empate técnico. No leas ruido como ranking.
- Portfolio de 100€: frecuencia 15-60 ops/año se mide AGREGADA; 3-5 activos
  máximo por nocional mínimo; el spread importa más que la comisión en órdenes
  de 10-30 USDT.
- Sé honesto: datos inferidos se declaran; PENDIENTE se declara; el fracaso es
  un resultado válido. Esto es experimentación iterativa.
