import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Importar función de cálculo de indicadores
# Usamos importación diferida o local si hay riesgo de ciclo, pero aquí debería estar bien
try:
    from ..indicators import calculate_indicator_and_signals
except ImportError:
    # Fallback si se ejecuta como script suelto sin contexto de paquete
    pass

def create_interactive_trade_chart(
    df: pd.DataFrame, 
    trades_df: pd.DataFrame, 
    title: str = "Trade Analysis", 
    filename: str = "chart.html",
    indicators: Optional[List[Dict[str, Any]]] = None
):
    """
    Genera un gráfico interactivo usando Bokeh y lo guarda en un archivo HTML.
    Soporta visualización de indicadores técnicos en paneles separados o overlay.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame OHLCV
    trades_df : pd.DataFrame
        DataFrame de trades
    title : str
        Título del gráfico
    filename : str
        Ruta donde guardar el archivo HTML
    indicators : list of dict, optional
        Lista de configuraciones de indicadores para visualizar.
        Ej: [{'indicator': 'rsi', 'params': {'period': 14}}]
    """
    try:
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, Span, CustomJSTickFormatter, Legend
        from bokeh.layouts import gridplot
        from bokeh.palettes import Category10
    except ImportError:
        return None

    # Preparar datos para velas
    df_viz = df.copy()
    df_viz['date'] = df_viz.index
    
    # Detectar columnas
    open_col = 'Open' if 'Open' in df_viz.columns else 'open'
    close_col = 'Close' if 'Close' in df_viz.columns else 'close'
    high_col = 'High' if 'High' in df_viz.columns else 'high'
    low_col = 'Low' if 'Low' in df_viz.columns else 'low'

    # --- CALCULAR INDICADORES ---
    indicator_plots_config = [] # Lista de configs para crear plots adicionales
    
    if indicators:
        for i, ind_config in enumerate(indicators):
            ind_name = ind_config['indicator']
            ind_params = ind_config.get('params', {})
            
            # Calcular indicador en copia temporal
            try:
                temp_df = calculate_indicator_and_signals(df_viz.copy(), ind_name, ind_params, inplace=False, use_cache=True)
                
                # Identificar columnas a graficar y tipo de panel
                # Mapeo de indicadores conocidos
                if ind_name == 'rsi':
                    col_name = f'RSI_{i}'
                    df_viz[col_name] = temp_df['indicator_value']
                    indicator_plots_config.append({
                        'type': 'panel', 'title': f'RSI ({ind_params.get("period", 14)})',
                        'series': [{'col': col_name, 'color': 'purple'}],
                        'lines': [30, 70], 'y_range': (0, 100)
                    })
                elif ind_name == 'macd':
                    df_viz[f'MACD_{i}'] = temp_df['macd']
                    df_viz[f'MACD_S_{i}'] = temp_df['macd_signal_line']
                    df_viz[f'MACD_H_{i}'] = temp_df['macd_histogram']
                    indicator_plots_config.append({
                        'type': 'panel', 'title': 'MACD',
                        'series': [
                            {'col': f'MACD_{i}', 'color': 'blue', 'label': 'MACD'},
                            {'col': f'MACD_S_{i}', 'color': 'orange', 'label': 'Signal'}
                        ],
                        'hist': {'col': f'MACD_H_{i}', 'color': 'gray'}
                    })
                elif ind_name == 'stoch':
                    df_viz[f'K_{i}'] = temp_df['stoch_k']
                    df_viz[f'D_{i}'] = temp_df['stoch_d']
                    indicator_plots_config.append({
                        'type': 'panel', 'title': 'Stochastic',
                        'series': [
                            {'col': f'K_{i}', 'color': 'blue', 'label': '%K'},
                            {'col': f'D_{i}', 'color': 'orange', 'label': '%D'}
                        ],
                        'lines': [20, 80], 'y_range': (0, 100)
                    })
                elif ind_name == 'adx':
                    df_viz[f'ADX_{i}'] = temp_df['adx']
                    df_viz[f'DI+_{i}'] = temp_df['adx_plus']
                    df_viz[f'DI-_{i}'] = temp_df['adx_minus']
                    indicator_plots_config.append({
                        'type': 'panel', 'title': 'ADX',
                        'series': [
                            {'col': f'ADX_{i}', 'color': 'black', 'label': 'ADX'},
                            {'col': f'DI+_{i}', 'color': 'green', 'label': 'DI+'},
                            {'col': f'DI-_{i}', 'color': 'red', 'label': 'DI-'}
                        ],
                        'lines': [25]
                    })
                elif ind_name == 'bbands': # Ejemplo overlay (si estuviera implementado)
                    # Asumiendo que calculate_indicator retorna upper/lower/mid
                    pass 
                elif 'indicator_value' in temp_df.columns:
                    # Genérico (CCI, WILLR, MOM, ROC, etc.)
                    col_name = f'{ind_name.upper()}_{i}'
                    df_viz[col_name] = temp_df['indicator_value']
                    
                    # Configuración por defecto para osciladores genéricos
                    lines = []
                    y_range = None
                    if ind_name == 'willr': lines = [-20, -80]; y_range = (-100, 0)
                    if ind_name == 'cci': lines = [-100, 100]
                    
                    indicator_plots_config.append({
                        'type': 'panel', 'title': f'{ind_name.upper()}',
                        'series': [{'col': col_name, 'color': 'blue'}],
                        'lines': lines, 'y_range': y_range
                    })
                    
            except Exception as e:
                print(f"⚠️ Error calculando indicador {ind_name} para visualización: {e}")

    # --- CREAR GRÁFICOS ---
    
    inc = df_viz[close_col] > df_viz[open_col]
    dec = df_viz[open_col] > df_viz[close_col]
    
    # Calcular ancho de velas
    if len(df_viz) > 1:
        min_diff = df_viz['date'].diff().min()
        w = min_diff.total_seconds() * 1000 * 0.8 # pyright: ignore[reportAttributeAccessIssue]
    else:
        w = 60 * 60 * 1000

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save,crosshair"

    # 1. Gráfico Principal (Precios)
    p_main = figure(x_axis_type="datetime", tools=TOOLS, width=1200, height=500, title=title,
               background_fill_color="#1e1e1e", border_fill_color="#1e1e1e")
    
    p_main.grid.grid_line_alpha = 0.3
    p_main.title.text_color = "white" # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    p_main.xaxis.axis_label_text_color = "white"
    p_main.yaxis.axis_label_text_color = "white"
    p_main.xaxis.major_label_text_color = "white"
    p_main.yaxis.major_label_text_color = "white"
    
    # Formato eje Y
    p_main.yaxis.formatter = CustomJSTickFormatter(code="""
        var abs_tick = Math.abs(tick);
        if (abs_tick >= 1000000) { return (tick / 1000000).toFixed(2).replace(/\.?0+$/, '') + 'M'; }
        else if (abs_tick >= 1000) { return (tick / 1000).toFixed(2).replace(/\.?0+$/, '') + 'K'; }
        else if (abs_tick < 0.0001 && abs_tick > 0) { return tick.toExponential(2); }
        else { return parseFloat(tick.toFixed(6)).toString(); }
    """)

    # Velas
    p_main.segment(df_viz.date, df_viz[high_col], df_viz.date, df_viz[low_col], color="white") # pyright: ignore[reportArgumentType]
    p_main.vbar(df_viz.date[inc], w, df_viz[open_col][inc], df_viz[close_col][inc], fill_color="#26a69a", line_color="#26a69a") # pyright: ignore[reportArgumentType]
    p_main.vbar(df_viz.date[dec], w, df_viz[open_col][dec], df_viz[close_col][dec], fill_color="#ef5350", line_color="#ef5350") # pyright: ignore[reportArgumentType]

    # Trades en gráfico principal
    if not trades_df.empty:
        # Longs
        long_entries = trades_df[trades_df['type'] == 'long']
        if not long_entries.empty:
            p_main.scatter(long_entries['entry_time'], long_entries['entry_price'], size=10, color="cyan", marker="triangle", legend_label="Long Entry") # pyright: ignore[reportArgumentType]
        
        # Shorts
        short_entries = trades_df[trades_df['type'] == 'short']
        if not short_entries.empty:
            p_main.scatter(short_entries['entry_time'], short_entries['entry_price'], size=10, color="orange", marker="inverted_triangle", legend_label="Short Entry") # pyright: ignore[reportArgumentType]

        # Exits
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        if not wins.empty:
            p_main.scatter(wins['exit_time'], wins['exit_price'], size=8, color="lime", marker="circle", legend_label="Exit (Win)") # pyright: ignore[reportArgumentType]
        if not losses.empty:
            p_main.scatter(losses['exit_time'], losses['exit_price'], size=8, color="red", marker="circle", legend_label="Exit (Loss)") # pyright: ignore[reportArgumentType]

    p_main.legend.location = "top_left"
    p_main.legend.click_policy = "hide"
    p_main.legend.label_text_color = "white"
    p_main.legend.background_fill_alpha = 0.5

    # 2. Paneles de Indicadores
    plots = [p_main]
    
    for ind_plot in indicator_plots_config:
        if ind_plot['type'] == 'panel':
            # Crear nuevo panel
            p_ind = figure(x_axis_type="datetime", tools=TOOLS, width=1200, height=200, 
                          x_range=p_main.x_range, # Vincular eje X
                          background_fill_color="#1e1e1e", border_fill_color="#1e1e1e",
                          title=ind_plot['title'])
            
            p_ind.grid.grid_line_alpha = 0.3
            p_ind.title.text_color = "white" # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            p_ind.xaxis.axis_label_text_color = "white"
            p_ind.yaxis.axis_label_text_color = "white"
            p_ind.xaxis.major_label_text_color = "white"
            p_ind.yaxis.major_label_text_color = "white"
            
            # Líneas de referencia (ej: 30/70 en RSI)
            if 'lines' in ind_plot:
                for y_val in ind_plot['lines']:
                    hline = Span(location=y_val, dimension='width', line_color='white', line_dash='dashed', line_width=1)
                    p_ind.add_layout(hline)
            
            # Rango Y fijo si se especifica
            if 'y_range' in ind_plot and ind_plot['y_range']:
                p_ind.y_range.start = ind_plot['y_range'][0] # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
                p_ind.y_range.end = ind_plot['y_range'][1] # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            
            # Histograma (ej: MACD)
            if 'hist' in ind_plot:
                hist_cfg = ind_plot['hist']
                p_ind.vbar(df_viz.date, w, 0, df_viz[hist_cfg['col']], color=hist_cfg['color'], alpha=0.5) # pyright: ignore[reportArgumentType]
            
            # Series de líneas
            for series in ind_plot['series']:
                label = series.get('label', series['col'])
                p_ind.line(df_viz.date, df_viz[series['col']], color=series['color'], line_width=2, legend_label=label) # pyright: ignore[reportArgumentType]
            
            p_ind.legend.label_text_color = "white"
            p_ind.legend.background_fill_alpha = 0.5
            p_ind.legend.click_policy = "hide"
            
            plots.append(p_ind)

    # Combinar en grid
    grid = gridplot([[p] for p in plots], toolbar_location="right")

    # Guardar
    output_file(filename, title=title, mode='inline')
    save(grid)
    
    return filename
