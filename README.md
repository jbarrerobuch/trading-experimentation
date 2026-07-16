# GammaNeutral: Trading Strategy Framework

GammaNeutral es un framework modular en Python diseñado para el desarrollo, backtesting, optimización y análisis de estrategias de trading algorítmico. Su arquitectura facilita la experimentación rápida con indicadores técnicos y combinaciones de estrategias utilizando configuraciones YAML y seguimiento de experimentos con MLflow.

## 🚀 Características Principales

*   **Gestión de Datos**: Carga y procesamiento eficiente de datos OHLCV (Open-High-Low-Close-Volume) en formato Parquet. Soporte para múltiples timeframes.
*   **Definición Declarativa**: Las estrategias se definen en archivos YAML, permitiendo ajustar parámetros e indicadores sin modificar el código.
*   **Backtesting Vectorizado**: Motor de backtesting rápido basado en pandas y numpy para evaluar el rendimiento histórico.
*   **Optimización Avanzada**:
    *   **Grid Search**: Búsqueda exhaustiva de combinaciones de parámetros.
    *   **Batch Processing**: Ejecución de optimizaciones en lotes grandes con capacidad de reanudación (checkpoints).
*   **Integración con MLflow**: Seguimiento detallado de experimentos, parámetros y métricas.
*   **Análisis y Visualización**: Generación automática de gráficos de rendimiento, heatmaps de correlación y estadísticas de trading.

## 📂 Estructura del Proyecto

El proyecto sigue una estructura modular organizada:

*   `src/trading_strategy/`: Núcleo del framework. Contiene la lógica de carga de datos, cálculo de indicadores, backtesting y utilidades.
*   `strategies/`: Directorio de configuraciones. Aquí residen los archivos YAML que definen las estrategias (ej. MACD, CCI, combinaciones).
*   `examples/`: Scripts de ejemplo que demuestran el flujo de trabajo típico (uso básico, optimización de portafolio, visualización).
*   `data/`: Almacenamiento de datos de mercado (`market/`) y artefactos generados (`img/`, `stats/`, `final_results/`).
*   `tests/`: Pruebas unitarias y de integración.

## 🛠️ Instalación

1.  Clona el repositorio:
    ```bash
    git clone https://github.com/jbarrerobuch/trading-experimentation.git
    cd trading-experimentation
    ```

2.  Instala las dependencias necesarias:
    ```bash
    pip install -r requirements.txt
    ```

    *Dependencias clave: pandas, numpy, pandas-ta, mlflow, matplotlib, seaborn, ccxt.*

## ⚡ Guía de Inicio Rápido

El script `examples/basic_usage.py` ilustra el flujo completo de trabajo.

1.  **Descargar Datos**: Asegúrate de tener datos históricos en `data/market`. Puedes usar las herramientas del framework (basadas en `ccxt`) para descargar datos de exchanges.
2.  **Ejecutar Ejemplo Básico**:
    ```bash
    python examples/basic_usage.py
    ```
    
    Este script realizará lo siguiente:
    1.  Cargará los datos de mercado.
    2.  Cargará las definiciones de estrategia desde `strategies/`.
    3.  Ejecutar un Grid Search para optimizar parámetros.
    4.  Visualizar los resultados y exportar las mejores estrategias a `data/stats`.

## ⚙️ Configuración de Estrategias

Las estrategias se definen en archivos YAML dentro de la carpeta `strategies/`. Ejemplo simplificado:

```yaml
name: "Estrategia_MACD_Ejemplo"
type: "individual"  # o 'combo'
indicator:
  name: "macd"
  params:
    fast: 12
    slow: 26
    signal: 9
params_grid:
  fast: [8, 12, 16]
  slow: [21, 26, 30]
# ... otros parámetros de gestión de riesgo
```

## 📊 MLflow

El proyecto utiliza MLflow para registrar experimentos. Para ver la interfaz gráfica de resultados:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Luego navega a `http://127.0.0.1:5000` en tu navegador.

## 🧪 Tests (WIP)

Para ejecutar las pruebas y asegurar la integridad del framework:

```bash
pytest tests/ -v
```
