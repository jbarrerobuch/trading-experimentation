"""
Módulo de Gestión de Portafolios y Combinación de Estrategias
Permite combinar múltiples estrategias y calcular métricas de portafolio.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import combinations
from .indicators import calculate_indicator_and_signals, combine_indicator_signals
from .constants import COL_SIGNAL, COL_OPEN, COL_CLOSE

class PortfolioManager:
    def __init__(self, market_data: pd.DataFrame, initial_capital: float = 10000.0,
                 commission: float = 0.001, slippage: float = 0.0001):
        """
        Inicializa el gestor de portafolios.
        
        Parameters:
        -----------
        market_data : pd.DataFrame
            DataFrame con datos de mercado. Debe contener 'returns' (o se calculará).
        initial_capital : float
            Capital inicial (para escalado, aunque en vectorizado se usa retorno porcentual).
        commission : float
            Costo por operación (tasa, ej: 0.001 para 0.1%).
        slippage : float
            Deslizamiento estimado (tasa).
        """
        self.market_data = market_data.copy()
        if 'returns' not in self.market_data.columns:
            self.market_data['returns'] = self.market_data[COL_CLOSE].pct_change()
            
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Almacena metadatos de estrategias
        self.strategies: Dict[str, Dict[str, Any]] = {}
        
        # Almacena vectores de retorno por estrategia
        self.strategy_returns: pd.DataFrame = pd.DataFrame(index=self.market_data.index)
        
    def add_strategy(self, 
                     name: str, 
                     indicator: Optional[str] = None, 
                     params: Dict[str, Any] = {}, 
                     position_type: str = 'long', 
                     indicators_combo: Optional[List[Dict[str, Any]]] = None, 
                     combination_method: str = 'AND',
                     use_next_open: bool = True):
        """
        Calcula y almacena el vector de retornos de una estrategia.
        """
        returns_vector = self._calculate_strategy_returns(
            indicator, params, position_type, indicators_combo, combination_method, use_next_open
        )
        
        self.strategy_returns[name] = returns_vector
        self.strategies[name] = {
            'indicator': indicator,
            'params': params,
            'position_type': position_type,
            'indicators_combo': indicators_combo,
            'combination_method': combination_method
        }
        
    def _calculate_strategy_returns(self, indicator, params, position_type, 
                                  indicators_combo, combination_method, use_next_open) -> pd.Series:
        """
        Genera el vector de retornos neto de costos para una configuración dada.
        """
        # Trabajamos sobre una copia slice para velocidad
        df = self.market_data.copy()
        
        # 1. Calcular Señal
        if indicators_combo is not None:
            df = combine_indicator_signals(df, indicators_combo, combination_method, inplace=True)
        else:
            if indicator is None:
                return pd.Series(0.0, index=df.index)
            df = calculate_indicator_and_signals(df, indicator, params, inplace=True)
            
        if COL_SIGNAL not in df.columns:
            return pd.Series(0.0, index=df.index)
            
        # 2. Determinar Posición
        raw_signal = df[COL_SIGNAL].fillna(0)
        
        if position_type == 'long':
            target_position = np.where(raw_signal > 0, 1, 0)
        elif position_type == 'short':
            target_position = np.where(raw_signal < 0, -1, 0)
        else: # both
            target_position = raw_signal.values
            
        # 3. Alinear ejecución
        if use_next_open:
            # Shift 1: La decisión tomada en T se ejecuta en T+1 (Open)
            # La posición efectiva en el candle T+1 es 'exec_position'
            exec_position = pd.Series(target_position, index=df.index).shift(1).fillna(0)
            
            # El retorno de la estrategia en T+1 depende de price change CLOSE(T) -> CLOSE(T+1)
            # Si entramos en Open de T+1, el retorno 'real' de la barra T+1 es (Close(T+1) - Open(T+1)) / Open(T+1)
            # Pero nuestro dataset suele tener retornos Close-to-Close.
            # Aproximación vectorial estándar: exec_pos[T] * (Close[T]-Close[T-1])/Close[T-1]
            # Esto asume que entramos al Close de T-1. 
            # Si usamos 'use_next_open', deberíamos usar el retorno Open-to-Close de esa vela para la primera vela,
            # pero para mantener consistencia con backtesting.py (que usa entry_price=Open), 
            # usaremos la aproximación Close-to-Close desplazada.
            
            # Simplificación: Usamos Close-to-Close returns multiplicados por la posición que traemos "desde ayer" (o entramos hoy Open)
            # Nota: Esto puede diferir pips del backtest por eventos intra-vela, pero es sólido para correlaciones.
            pass
        else:
            exec_position = pd.Series(target_position, index=df.index)
            
        # 4. Calcular Retornos Brutos
        # Retorno de Estrategia = Retorno Mercado * Posición
        market_returns = df['returns']
        strategy_gross_returns = market_returns * exec_position
        
        # 5. Calcular Costos
        # Cambio de posición implica trade
        pos_diff = exec_position.diff().abs().fillna(0)
        # Costo = (Commission + Slippage) por cada unidad de cambio de posición (0->1 es 100%, 1->-1 es 200% rotación)
        trading_costs = pos_diff * (self.commission + self.slippage)
        
        strategy_net_returns = strategy_gross_returns - trading_costs
        return strategy_net_returns

    def calculate_correlation_matrix(self):
        """Devuelve la matriz de correlación de los retornos de las estrategias."""
        return self.strategy_returns.corr()
    
    def simulate_portfolio(self, strategy_names: List[str], weights: Optional[List[float]] = None) -> pd.Series:
        """
        Simula retornos combinados (equally weighted por defecto).
        """
        subset = self.strategy_returns[strategy_names]
        
        if weights is None:
            # Equal weight
            combined_ret = subset.mean(axis=1) # Promedio de retornos (rebalanceo diario implícito)
        else:
            # Weighted
            if len(weights) != len(strategy_names):
                raise ValueError("Longitud de weights debe coincidir con strategy_names")
            # Normalizar weights
            w = np.array(weights) / np.sum(weights)
            combined_ret = subset.dot(w)
            
        return combined_ret
        
    def calculate_portfolio_metrics(self, returns_series: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Calcula métricas vectorizadas para el portafolio."""
        if returns_series.empty or returns_series.isna().all():
            return {}
            
        # Limpieza
        returns = returns_series.fillna(0)
        
        n_periods = len(returns)
        if n_periods < 2:
            return {}
            
        # Acumulado
        total_return = (1 + returns).prod() - 1 # pyright: ignore[reportOperatorIssue]
        
        # Annualization factor (asumiendo datos horarios -> 24*365 = 8760, diarios -> 365 or 252)
        # Inferir frecuencia es difícil solo con vector, asumiremos hourly crypto (24*365) si len > 10000, else daily?
        # Mejor usar un parametro, pero por defecto usaremos Crypto 1h (muy comun en este repo)
        annual_factor = 365 * 24 
        
        # CAGR
        cagr = (1 + total_return) ** (annual_factor / n_periods) - 1 if total_return > -1 else -1 # pyright: ignore[reportOperatorIssue]

        # Volatility
        volatility = returns.std() * np.sqrt(annual_factor)
        
        # Sharpe
        excess_returns = returns - (risk_free_rate / annual_factor)
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(annual_factor) if returns.std() != 0 else 0
        
        # Sortino
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(annual_factor)
        sortino = (excess_returns.mean() / downside_std) * np.sqrt(annual_factor) if downside_std != 0 else 0
        
        # Max Drawdown
        cum_returns = (1 + returns).cumprod()
        # Fix: Asegurar que el pico inicial sea al menos 1.0 para capturar DD desde el inicio
        peak = np.maximum(cum_returns.cummax(), 1.0)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Explicit type conversion to ensure keys match Dict[str, float]
        return {
            'total_return': float(total_return), # pyright: ignore[reportArgumentType]
            'cagr': float(cagr), # pyright: ignore[reportArgumentType]
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'calmar_ratio': float(calmar)
        }

    def find_best_combinations(self, min_k=2, max_k=3, sort_by='calmar_ratio', top_n=10):
        """
        Busca fuerza bruta las mejores combinaciones de estrategias.
        """
        all_names = list(self.strategies.keys())
        results = []
        
        import tqdm # Opcional si está disponible
        
        for k in range(min_k, max_k + 1):
            for combo in combinations(all_names, k):
                # Calcular portafolio equal-weight
                try:
                    combined_ret = self.simulate_portfolio(list(combo))
                    metrics = self.calculate_portfolio_metrics(combined_ret)
                    metrics['strategies'] = combo # pyright: ignore[reportArgumentType]
                    metrics['n_strategies'] = k
                    results.append(metrics)
                except Exception as e:
                    continue
                    
        # Convertir a DF y ordenar
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(by=sort_by, ascending=False).head(top_n)
            
        return results_df

