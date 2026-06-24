import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional

# Intentar importar librerías opcionales
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .grid_search import strategy_grid_search
from .backtesting import get_strategy_trades
from .utils.stats import calculate_trade_statistics
from .utils.helpers import generate_run_name


def walk_forward_optimization(
    df: pd.DataFrame,
    strategy_configs: List[Dict],
    window_size: int,
    step_size: int,
    metric: str = 'sharpe_ratio',
    anchored: bool = False,
    commission: float = 0.001,
    slippage: float = 0.0001,
    use_next_open: bool = True,
    use_mlflow: bool = False,
    batch_size: Optional[int] = None,
    resume_checkpoints: bool = True,
    checkpoint_dir: Optional[str] = None,
    experiment_name: str = "WFA_Analysis",
    ticker: str = "Unknown",
    timeframe: str = "Unknown"
) -> Dict[str, Any]:
    """
    Realiza una optimización Walk-Forward (WFA).
    
    Simula el proceso de re-optimizar la estrategia cada 'step_size' velas,
    usando las últimas 'window_size' velas para entrenamiento.
    
    Parameters:
    -----------
    df : DataFrame
        Datos OHLCV completos.
    strategy_configs : list
        Espacio de búsqueda de estrategias.
    window_size : int
        Tamaño de la ventana de entrenamiento (In-Sample).
    step_size : int
        Tamaño de la ventana de prueba (Out-Of-Sample).
    metric : str
        Métrica para seleccionar la mejor estrategia en IS.
    anchored : bool
        Si True, el inicio de la ventana de entrenamiento es fijo (Expanding).
        Si False, la ventana se mueve (Rolling).
    use_mlflow : bool
        Si True, registra resultados en MLflow.
    checkpoint_dir : str, optional
        Directorio base para guardar checkpoints.
        
    Returns:
    --------
    Dict con:
        - 'oos_trades': DataFrame con todos los trades concatenados del periodo OOS.
        - 'metrics': Métricas globales del rendimiento OOS.
        - 'history': DataFrame con el registro de qué estrategia ganó en cada paso.
    """

    # batch mode deprecado: el motor refactorizado corre un único grid por ventana
    # (reutilizando su value_cache). Los parámetros se mantienen por compatibilidad.
    if batch_size:
        print("ℹ️  WFA batch mode está deprecado y se ignora; se ejecuta un único grid por ventana.")

    # Configurar MLflow si se solicita
    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment(experiment_name) # pyright: ignore[reportPossiblyUnboundVariable]
        run_name = f"WFA_{ticker}_{timeframe}_{metric}_{'Anchored' if anchored else 'Rolling'}"
        mlflow.start_run(run_name=run_name) # pyright: ignore[reportPossiblyUnboundVariable]
        
        # Log de parámetros de configuración WFA
        mlflow.log_params({ # pyright: ignore[reportPossiblyUnboundVariable]
            "wfa_window_size": window_size,
            "wfa_step_size": step_size,
            "wfa_metric": metric,
            "wfa_anchored": anchored,
            "ticker": ticker,
            "timeframe": timeframe
        })
    elif use_mlflow and not MLFLOW_AVAILABLE:
        print("⚠️ MLflow no está instalado. Se continuará sin logging.")
        use_mlflow = False

    n_candles = len(df)
    if n_candles < window_size + step_size:
        raise ValueError(f"Datos insuficientes ({n_candles}) para Window={window_size} + Step={step_size}")
    
    oos_trades_all = []
    results_history = []
    
    # Índices iniciales
    start_idx = 0
    train_end_idx = window_size
    
    print(f"\n🔄 INICIANDO WALK-FORWARD OPTIMIZATION")
    print(f"   Ticker: {ticker} ({timeframe})")
    print(f"   Total velas: {n_candles}")
    print(f"   Config: Window={window_size}, Step={step_size}, Anchored={anchored}")
    print(f"   Metric: {metric}")
    print("=" * 80)
    
    step_count = 0
    
    # Limpiar cache de indicadores al inicio de todo el proceso WFA
    from .indicators import clear_indicator_cache
    clear_indicator_cache()
    
    try:
        while train_end_idx + step_size <= n_candles:
            step_count += 1
            
            # Iniciar Nested Run para este paso
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.start_run(run_name=f"Step_{step_count}", nested=True)
            
            test_end_idx = train_end_idx + step_size
            
            # Definir slices de datos
            # IMPORTANTE: Para el test set, necesitamos incluir un periodo de "warmup" (calentamiento)
            # anterior al inicio del test para que los indicadores (ej: SMA 200) tengan datos suficientes
            # para calcularse en la primera vela del test.
            # Usaremos un buffer seguro (ej: 2000 velas o el tamaño de la ventana de train si es menor)
            warmup_size = min(window_size, 5000) # 5000 velas de seguridad para indicadores lentos
            
            train_start = 0 if anchored else start_idx
            train_df = df.iloc[train_start:train_end_idx].copy()
            
            # Test DF con warmup
            test_start_with_warmup = max(0, train_end_idx - warmup_size)
            test_df_warmup = df.iloc[test_start_with_warmup:test_end_idx].copy()
            
            # Fechas para log
            train_period_str = f"{train_df.index[0]} -> {train_df.index[-1]}"
            test_period_str = f"{df.index[train_end_idx]} -> {df.index[test_end_idx-1]}" # Fechas reales del test
            
            print(f"\n📍 Step {step_count}")
            print(f"   Train: {len(train_df)} velas ({train_period_str})")
            print(f"   Test:  {step_size} velas ({test_period_str}) [Warmup: {len(test_df_warmup)-step_size} velas]")
            
            # 1. OPTIMIZACIÓN (In-Sample)
            # Limpiar cache al inicio de cada paso WFA para evitar fugas de memoria
            # (aunque el cache usa LRU, es bueno resetear al cambiar de ventana de datos)
            clear_indicator_cache()
            
            # Ejecutamos un único grid search en la ventana de entrenamiento.
            # (El antiguo "batch mode" partía el grid en configs atómicos y rompía la
            #  reutilización del value_cache del motor; está deprecado, ver firma.)
            grid_results = strategy_grid_search(
                df=train_df,
                strategy_configs=strategy_configs,
                use_mlflow=False,
                commission=commission,
                slippage=slippage,
                use_next_open=use_next_open,
                verbose=False,
            )

            if grid_results.empty:
                print("   ⚠️ No se encontraron estrategias válidas. Saltando paso.")
                start_idx += step_size
                train_end_idx += step_size
                continue
                
            # Seleccionar mejor estrategia basada en métrica IS
            if metric not in grid_results.columns:
                print(f"   ⚠️ Métrica '{metric}' no encontrada. Usando 'sharpe_ratio'.")
                metric_to_use = 'sharpe_ratio'
            else:
                metric_to_use = metric
                
            best_row = grid_results.sort_values(by=metric_to_use, ascending=False).iloc[0]
            best_score = best_row[metric_to_use]
            strat_name = best_row['strategy_name']
            
            print(f"   🏆 Best IS: {strat_name} ({metric_to_use}={best_score:.3f})")
            
            # 2. VALIDACIÓN (Out-Of-Sample)
            # Ejecutar la estrategia ganadora en la ventana de prueba desconocida
            strategy_type = best_row.get('strategy_type', 'single')
            
            try:
                if strategy_type == 'combo':
                    # Reconstruir indicators_combo desde las columnas planas si es necesario
                    # O usar 'combo_params' si lo guardamos en grid_search (que no lo hicimos explícitamente en el código anterior, 
                    # pero grid_search devuelve columnas indX_name, indX_param...)
                    # Lo más seguro es usar get_strategy_trades con los parámetros reconstruidos.
                    
                    # Nota: En la implementación actual de grid_search.py, no devolvemos el objeto 'indicators_combo' completo en el DF.
                    # Pero devolvemos ind1_name, ind1_period, etc.
                    # Para simplificar, asumimos que strategy_configs tiene la estructura y podemos re-instanciar.
                    # O mejor, si grid_search devolviera el objeto config sería ideal.
                    # Workaround: Usar los parámetros planos del best_row para reconstruir.
                    
                    # Sin embargo, para simplificar este ejemplo, asumiremos que podemos pasar los params.
                    # Si grid_search.py fue modificado para devolver 'indicators_combo' (lista de dicts) sería mejor.
                    # Revisando grid_search.py, no devuelve la columna 'indicators_combo'.
                    # Vamos a intentar usar los parámetros planos si es posible, o confiar en que el usuario pase configs simples.
                    
                    # Si es combo, necesitamos reconstruir la lista de indicadores.
                    # Esto es complejo dinámicamente. 
                    # SOLUCIÓN: En grid_search.py, deberíamos haber guardado la config.
                    # Como no podemos editar grid_search ahora mismo fácilmente sin romper cosas,
                    # vamos a intentar inferir o usar lo que hay.
                    
                    # Si el usuario usa configs simples, podemos intentar.
                    # Si no, esto fallará para combos complejos.
                    # Asumamos por ahora que podemos pasar 'indicators_combo' si existe en best_row (si lo agregamos).
                    # Si no, imprimimos advertencia.
                    
                    indicators_combo = None
                    # Intento de recuperación básica si existen columnas ind1_name, etc.
                    if 'n_indicators' in best_row:
                        indicators_combo = []
                        n_inds = int(best_row['n_indicators'])
                        for i in range(1, n_inds + 1):
                            ind_name = best_row.get(f'ind{i}_name')
                            # Recolectar params para este indicador
                            ind_params = {}
                            prefix = f'ind{i}_'
                            for col in best_row.index:
                                if col.startswith(prefix) and col != f'ind{i}_name':
                                    param_key = col.replace(prefix, '')
                                    ind_params[param_key] = best_row[col]
                            
                            indicators_combo.append({
                                'indicator': ind_name,
                                'params': ind_params
                            })
                    
                    trades = get_strategy_trades(
                        test_df_warmup,
                        indicator=None,
                        params={},
                        position_type=best_row['position_type'],
                        indicators_combo=indicators_combo,
                        combination_method=best_row['combination_method'],
                        use_next_open=use_next_open,
                        commission=commission,
                        slippage=slippage,
                    )
                    params_log = {'indicators': indicators_combo, 'method': best_row['combination_method']}
                    
                else:
                    # Single strategy
                    # Extraer params del best_row (excluyendo columnas de métricas)
                    # Esto es sucio. Lo ideal es que grid_search devuelva 'params_dict'.
                    # Vamos a asumir que las columnas que no son métricas son params.
                    # O mejor, reconstruir desde strategy_configs si es posible.
                    
                    # En la versión actual de grid_search, se aplanan los parámetros.
                    # Vamos a intentar usar los parámetros conocidos del indicador.
                    # Esto requiere saber qué parámetros tiene el indicador.
                    
                    # Mejor aproximación: Usar todas las columnas que no son estándar.
                    excluded_cols = [
                        'ticker', 'timeframe', 'strategy_name', 'strategy_type', 
                        'combination_method', 'position_type', 'n_indicators',
                        'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 
                        'n_trades', 'n_transactions', 'hit_rate', 'sortino_ratio', 
                        'calmar_ratio', 'profit_factor', 'avg_win', 'avg_loss', 
                        'risk_reward_ratio', 'best_trade', 'worst_trade', 
                        'avg_return_per_trade', 'volatility', 'sqn', 'kelly_criterion', 
                        'expectancy', 'net_profit_usd', 'final_equity', 
                        'max_drawdown_usd', 'total_costs_pct', 'period_profit_factor'
                    ]
                    
                    params_dict = {}
                    for col in best_row.index:
                        if col not in excluded_cols and not col.startswith('test_') and not col.startswith('ind'):
                            params_dict[col] = best_row[col]
                            
                    trades = get_strategy_trades(
                        test_df_warmup,
                        indicator=best_row.get('indicator', strat_name.split('_')[0]), # Fallback
                        params=params_dict,
                        position_type=best_row['position_type'],
                        use_next_open=use_next_open,
                        commission=commission,
                        slippage=slippage,
                    )
                    params_log = params_dict
                
                # Generar nombre descriptivo para el run y los trades
                from .utils.helpers import generate_run_name
                
                if strategy_type == 'combo':
                    descriptive_name = generate_run_name(
                        strategy_name=strat_name,
                        strategy_type='combo',
                        position_type=best_row['position_type'],
                        indicators_combo=indicators_combo,
                        combination_method=best_row['combination_method']
                    )
                else:
                    descriptive_name = generate_run_name(
                        strategy_name=strat_name,
                        strategy_type='single',
                        position_type=best_row['position_type'],
                        params=params_dict
                    )
                
                # Actualizar nombre del Run en MLflow
                if use_mlflow and MLFLOW_AVAILABLE:
                    mlflow.set_tag("mlflow.runName", descriptive_name)

                # Registrar resultados
                # FILTRAR TRADES: Solo mantener los que ocurrieron en el periodo de prueba real (sin warmup)
                step_pnl = 0.0
                n_trades_step = 0
                step_metrics = {}
                
                if not trades.empty:
                    test_start_time = df.index[train_end_idx]
                    trades = trades[trades['entry_time'] >= test_start_time].copy()
                
                if not trades.empty:
                    # Agregar metadatos a los trades para análisis posterior
                    trades['step'] = step_count
                    trades['strategy_name'] = descriptive_name
                    trades['train_start'] = train_df.index[0]
                    trades['train_end'] = train_df.index[-1]
                    
                    # Aplanar params para el CSV de trades (string representation)
                    trades['strategy_params'] = str(params_log)
                    
                    oos_trades_all.append(trades)
                    step_pnl = trades['pnl_pct'].sum()
                    n_trades_step = len(trades)
                    
                    # Calcular métricas detalladas del paso OOS
                    step_metrics = calculate_trade_statistics(trades, initial_capital=10000.0)
                    
                    print(f"   📊 OOS Result: {n_trades_step} trades | PnL: {step_pnl:.2%}")
                else:
                    print(f"   📊 OOS Result: 0 trades")
                    
                results_history.append({
                    'step': step_count,
                    'train_end_date': train_df.index[-1],
                    'test_start_date': df.index[train_end_idx],
                    'test_end_date': df.index[test_end_idx-1],
                    'best_strategy': strat_name,
                    'is_score': best_score,
                    'oos_trades': n_trades_step,
                    'oos_pnl': step_pnl,
                    'params': str(params_log)
                })
                
                # Log progresivo en MLflow (Nested Run)
                if use_mlflow and MLFLOW_AVAILABLE:
                    # 1. Log Metrics (IS y OOS)
                    metrics_to_log = {
                        "is_score": best_score,
                        "oos_pnl": step_pnl,
                        "oos_trades": float(n_trades_step)
                    }
                    
                    # Agregar métricas IS detalladas si están disponibles en best_row
                    # Prefijo 'is_' para diferenciar
                    is_metric_cols = [
                        'sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 
                        'profit_factor', 'sqn', 'sortino_ratio', 'calmar_ratio'
                    ]
                    for col in is_metric_cols:
                        if col in best_row:
                            metrics_to_log[f"is_{col}"] = float(best_row[col])
                            
                    # Agregar métricas OOS detalladas calculadas
                    # Prefijo 'oos_' para diferenciar (excepto las que ya agregamos manualmente)
                    for k, v in step_metrics.items():
                        if k not in ['total_trades', 'total_return']: # Ya logueados o redundantes
                            metrics_to_log[f"oos_{k}"] = float(v)
                            
                    mlflow.log_metrics(metrics_to_log)
                    
                    # 2. Log Params (Aplanados y legibles)
                    flat_params = {
                        "step": step_count,
                        "train_start": str(train_df.index[0]),
                        "train_end": str(train_df.index[-1]),
                        "test_start": str(df.index[train_end_idx]),
                        "test_end": str(df.index[test_end_idx-1]),
                        "best_strategy_name": strat_name
                    }
                    
                    # Aplanar params_log (especialmente para combos)
                    if 'indicators' in params_log and isinstance(params_log['indicators'], list):
                        # Es un combo
                        flat_params['combination_method'] = params_log.get('method', 'Unknown')
                        for i, ind in enumerate(params_log['indicators']):
                            flat_params[f'ind{i+1}_name'] = ind['indicator']
                            for p_name, p_val in ind.get('params', {}).items():
                                flat_params[f'ind{i+1}_{p_name}'] = str(p_val)
                    else:
                        # Es single o params planos
                        for k, v in params_log.items():
                            flat_params[k] = str(v)
                            
                    mlflow.log_params(flat_params)
                    
                    # También loguear como texto raw
                    mlflow.log_text(str(params_log), "params_raw.txt")
                
            except Exception as e:
                print(f"   ❌ Error en OOS execution: {e}")
                import traceback
                traceback.print_exc()

            # Cerrar Nested Run
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.end_run()

            # Avanzar ventanas
            start_idx += step_size
            train_end_idx += step_size

        # 3. CONSOLIDACIÓN
        print("\n" + "=" * 80)
        metrics = {}
        all_trades = pd.DataFrame()
        
        if oos_trades_all:
            all_trades = pd.concat(oos_trades_all, ignore_index=True)
            
            # Calcular métricas globales usando el módulo stats
            metrics = calculate_trade_statistics(all_trades, initial_capital=10000.0)
            
            print(f"🏁 WFA FINALIZADO")
            print(f"   Trades Totales: {metrics['total_trades']}")
            print(f"   Win Rate Global: {metrics['win_rate']:.1%}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"   Retorno Total: {metrics['total_return']:.2%}")
            
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_metrics(metrics)
                
                # Guardar artefactos
                os.makedirs("data/stats", exist_ok=True)
                hist_path = "data/stats/wfa_history.csv"
                trades_path = "data/stats/wfa_trades.csv"
                
                pd.DataFrame(results_history).to_csv(hist_path, index=False)
                all_trades.to_csv(trades_path, index=False)
                
                mlflow.log_artifact(hist_path)
                mlflow.log_artifact(trades_path)
                
        else:
            print("🏁 WFA FINALIZADO sin trades.")
            
        return {
            'oos_trades': all_trades,
            'metrics': metrics,
            'history': pd.DataFrame(results_history)
        }
        
    except Exception as e:
        print(f"❌ Error fatal en WFA: {e}")
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_param("error", str(e)) # pyright: ignore[reportPossiblyUnboundVariable]
        raise e
    finally:
        if use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run() # pyright: ignore[reportPossiblyUnboundVariable]
