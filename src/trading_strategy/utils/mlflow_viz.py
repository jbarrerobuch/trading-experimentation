import pandas as pd
import numpy as np

def create_interactive_trade_chart(df: pd.DataFrame, trades_df: pd.DataFrame, title: str = "Trade Analysis", filename: str = "chart.html"):
    """
    Genera un gráfico interactivo usando Bokeh y lo guarda en un archivo HTML.
    
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
    """
    try:
        from bokeh.plotting import figure, output_file, save
        from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, Span
        from bokeh.layouts import gridplot
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

    inc = df_viz[close_col] > df_viz[open_col]
    dec = df_viz[open_col] > df_viz[close_col]
    
    # Calcular ancho de velas (aprox 80% del intervalo mínimo)
    if len(df_viz) > 1:
        min_diff = df_viz['date'].diff().min()
        w = min_diff.total_seconds() * 1000 * 0.8 # pyright: ignore[reportAttributeAccessIssue] # en ms
    else:
        w = 60 * 60 * 1000 # default 1h

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    p = figure(x_axis_type="datetime", tools=TOOLS, width=1200, height=600, title=title,
               background_fill_color="#1e1e1e", border_fill_color="#1e1e1e")
    
    p.grid.grid_line_alpha = 0.3
    p.title.text_color = "white" # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    p.xaxis.axis_label_text_color = "white"
    p.yaxis.axis_label_text_color = "white"
    p.xaxis.major_label_text_color = "white"
    p.yaxis.major_label_text_color = "white"

    # Velas
    p.segment(df_viz.date, df_viz[high_col], df_viz.date, df_viz[low_col], color="white") # pyright: ignore[reportArgumentType]
    p.vbar(df_viz.date[inc], w, df_viz[open_col][inc], df_viz[close_col][inc], fill_color="#26a69a", line_color="#26a69a") # pyright: ignore[reportArgumentType]
    p.vbar(df_viz.date[dec], w, df_viz[open_col][dec], df_viz[close_col][dec], fill_color="#ef5350", line_color="#ef5350") # pyright: ignore[reportArgumentType]

    # Trades
    if not trades_df.empty:
        # Long Entries
        long_entries = trades_df[trades_df['type'] == 'long']
        if not long_entries.empty:
            p.scatter(long_entries['entry_time'], long_entries['entry_price'], size=10, color="cyan", marker="triangle", legend_label="Long Entry") # pyright: ignore[reportArgumentType]
            
        # Short Entries
        short_entries = trades_df[trades_df['type'] == 'short']
        if not short_entries.empty:
            p.scatter(short_entries['entry_time'], short_entries['entry_price'], size=10, color="orange", marker="inverted_triangle", legend_label="Short Entry") # pyright: ignore[reportArgumentType]

        # Exits (Wins/Losses)
        wins = trades_df[trades_df['pnl_pct'] > 0]
        losses = trades_df[trades_df['pnl_pct'] <= 0]
        
        if not wins.empty:
            p.scatter(wins['exit_time'], wins['exit_price'], size=8, color="lime", marker="circle", legend_label="Exit (Win)") # pyright: ignore[reportArgumentType]
        
        if not losses.empty:
            p.scatter(losses['exit_time'], losses['exit_price'], size=8, color="red", marker="circle", legend_label="Exit (Loss)") # pyright: ignore[reportArgumentType]

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.label_text_color = "white"
    p.legend.background_fill_alpha = 0.5

    # Guardar
    output_file(filename, title=title, mode='inline')
    save(p)
    
    return filename
