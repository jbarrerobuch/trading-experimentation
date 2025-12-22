"""
Módulo de visualización y exportación de resultados
Genera gráficos y exporta mejores estrategias
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_grid_search_results(results_df, save_path='../data/img/grid_search_results.png'):
    """
    Visualiza resultados del Grid Search en 6 gráficos
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame con resultados del grid search
    save_path : str
        Ruta donde guardar la imagen
    """
    if results_df.empty:
        print("⚠️  No hay resultados para visualizar")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Grid Search Results - Strategy Optimization', fontsize=16, fontweight='bold')
    
    # 1. Distribución de Sharpe Ratios
    axes[0, 0].hist(results_df['sharpe_ratio'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results_df['sharpe_ratio'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["sharpe_ratio"].mean():.2f}')
    axes[0, 0].axvline(results_df['sharpe_ratio'].median(), color='green', linestyle='--', label=f'Median: {results_df["sharpe_ratio"].median():.2f}')
    axes[0, 0].set_xlabel('Sharpe Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Sharpe Ratios')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Win Rate vs Sharpe Ratio
    axes[0, 1].scatter(results_df['win_rate'], results_df['sharpe_ratio'], alpha=0.6, s=50)
    axes[0, 1].set_xlabel('Win Rate')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].set_title('Win Rate vs Sharpe Ratio')
    axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Total Return vs Max Drawdown (Risk-Return)
    axes[0, 2].scatter(results_df['max_drawdown'], results_df['total_return'], alpha=0.6, s=50, c=results_df['sharpe_ratio'], cmap='RdYlGn')
    axes[0, 2].set_xlabel('Max Drawdown')
    axes[0, 2].set_ylabel('Total Return')
    axes[0, 2].set_title('Risk-Return Profile')
    axes[0, 2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 2].axvline(0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 2].grid(alpha=0.3)
    cbar = plt.colorbar(axes[0, 2].collections[0], ax=axes[0, 2])
    cbar.set_label('Sharpe Ratio')
    
    # 4. Top 10 Strategies
    top10 = results_df.nlargest(10, 'sharpe_ratio')
    strategy_labels = [f"{row['strategy_name'][:15]}\n{row.get('indicator', row.get('combination_method', 'N/A'))}" 
                      for _, row in top10.iterrows()]
    axes[1, 0].barh(range(len(top10)), top10['sharpe_ratio'], color='steelblue')
    axes[1, 0].set_yticks(range(len(top10)))
    axes[1, 0].set_yticklabels(strategy_labels, fontsize=8)
    axes[1, 0].set_xlabel('Sharpe Ratio')
    axes[1, 0].set_title('Top 10 Strategies by Sharpe Ratio')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].invert_yaxis()
    
    # 5. Profit Factor por tipo de posición
    if 'position_type' in results_df.columns:
        position_stats = results_df.groupby('position_type').agg({
            'sharpe_ratio': 'mean',
            'profit_factor': 'mean',
            'win_rate': 'mean'
        })
        
        x_pos = range(len(position_stats))
        axes[1, 1].bar([p - 0.2 for p in x_pos], position_stats['profit_factor'], width=0.4, label='Profit Factor', alpha=0.8)
        axes[1, 1].bar([p + 0.2 for p in x_pos], position_stats['sharpe_ratio'], width=0.4, label='Sharpe Ratio', alpha=0.8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(position_stats.index)
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Performance by Position Type')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    
    # 6. Sharpe Ratio vs Number of Trades
    axes[1, 2].scatter(results_df['n_trades'], results_df['sharpe_ratio'], alpha=0.6, s=50)
    axes[1, 2].set_xlabel('Number of Trades')
    axes[1, 2].set_ylabel('Sharpe Ratio')
    axes[1, 2].set_title('Sharpe Ratio vs Trade Frequency')
    axes[1, 2].axhline(0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 2].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada: {save_path}")
    
    plt.show()


def export_best_strategies(results_df, top_n=10, save_path='../data/stats/best_strategies.json'):
    """
    Exporta las mejores estrategias a JSON
    
    Parameters:
    -----------
    results_df : DataFrame
        DataFrame con resultados del grid search
    top_n : int
        Número de mejores estrategias a exportar
    save_path : str
        Ruta donde guardar el archivo JSON
    """
    if results_df.empty:
        print("⚠️  No hay resultados para exportar")
        return
    
    top_strategies = results_df.nlargest(top_n, 'sharpe_ratio')
    
    export_data = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_strategies_tested': len(results_df),
        'top_strategies': top_strategies.to_dict('records')
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"✓ Mejores estrategias exportadas: {save_path}")
    print(f"  Top {top_n} estrategias guardadas")


def plot_strategy_equity_curve(df, signal_column='signal', title='Strategy Equity Curve'):
    """
    Grafica la curva de equity de una estrategia
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame con columnas 'returns' y signal_column
    signal_column : str
        Nombre de la columna con señales de trading
    title : str
        Título del gráfico
    """
    if signal_column not in df.columns:
        print(f"⚠️  Columna '{signal_column}' no encontrada")
        return
    
    df = df.copy()
    # Usar future_ret+1 si existe, sino calcularlo
    if 'future_ret+1' in df.columns:
        target_col = 'future_ret+1'
    elif 'future_ret' in df.columns:
        target_col = 'future_ret'
    else:
        target_col = 'future_ret+1'
        df[target_col] = df['returns'].shift(-1)
    
    # Aplicar señales
    df['strategy_ret'] = 0
    mask = df[signal_column] != 0
    df.loc[mask, 'strategy_ret'] = df.loc[mask, target_col] * df.loc[mask, signal_column]
    
    # Calcular equity curves
    df['buy_hold_equity'] = (1 + df['returns']).cumprod()
    df['strategy_equity'] = (1 + df['strategy_ret']).cumprod()
    
    # Graficar
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['buy_hold_equity'], label='Buy & Hold', alpha=0.7, linewidth=2)
    plt.plot(df.index, df['strategy_equity'], label='Strategy', alpha=0.7, linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Métricas
    total_ret_bh = df['buy_hold_equity'].iloc[-1] - 1
    total_ret_strategy = df['strategy_equity'].iloc[-1] - 1
    
    print(f"\n📊 Performance Comparison:")
    print(f"Buy & Hold: {total_ret_bh:.2%}")
    print(f"Strategy:   {total_ret_strategy:.2%}")
    print(f"Outperformance: {total_ret_strategy - total_ret_bh:.2%}")
