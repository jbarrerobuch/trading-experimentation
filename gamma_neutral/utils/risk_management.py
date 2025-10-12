"""
Risk Management Module for gamma neutral strategy.

This module provides risk management functionality including position sizing,
exposure limits, and risk metrics calculation.
"""

from typing import Dict, List, Optional
import numpy as np


class RiskManager:
    """
    Manages risk for the gamma neutral strategy.
    
    Provides functionality for:
    - Position sizing based on risk parameters
    - Exposure limit monitoring
    - Risk metrics calculation (VaR, expected shortfall, etc.)
    """
    
    def __init__(
        self,
        max_portfolio_delta: float = 1.0,
        max_portfolio_gamma: float = 0.5,
        max_position_size: float = 100.0,
        max_notional_exposure: float = 1000000.0,
        var_confidence: float = 0.95
    ):
        """
        Initialize the risk manager.
        
        Args:
            max_portfolio_delta: Maximum allowed absolute portfolio delta
            max_portfolio_gamma: Maximum allowed absolute portfolio gamma
            max_position_size: Maximum size for a single position
            max_notional_exposure: Maximum total notional exposure
            var_confidence: Confidence level for VaR calculation
        """
        self.max_portfolio_delta = max_portfolio_delta
        self.max_portfolio_gamma = max_portfolio_gamma
        self.max_position_size = max_position_size
        self.max_notional_exposure = max_notional_exposure
        self.var_confidence = var_confidence
    
    def check_position_limits(
        self,
        position_size: float,
        current_notional: float,
        position_notional: float
    ) -> Dict[str, bool]:
        """
        Check if a position meets size and exposure limits.
        
        Args:
            position_size: Size of the position
            current_notional: Current total notional exposure
            position_notional: Notional value of new position
        
        Returns:
            Dictionary with limit check results
        """
        within_position_limit = abs(position_size) <= self.max_position_size
        within_notional_limit = (current_notional + abs(position_notional)) <= self.max_notional_exposure
        
        return {
            "within_position_limit": within_position_limit,
            "within_notional_limit": within_notional_limit,
            "passes_all_checks": within_position_limit and within_notional_limit,
            "max_position_size": self.max_position_size,
            "max_notional_exposure": self.max_notional_exposure,
        }
    
    def check_greeks_limits(
        self,
        portfolio_delta: float,
        portfolio_gamma: float
    ) -> Dict[str, bool]:
        """
        Check if portfolio Greeks are within limits.
        
        Args:
            portfolio_delta: Portfolio delta
            portfolio_gamma: Portfolio gamma
        
        Returns:
            Dictionary with limit check results
        """
        delta_within_limit = abs(portfolio_delta) <= self.max_portfolio_delta
        gamma_within_limit = abs(portfolio_gamma) <= self.max_portfolio_gamma
        
        return {
            "delta_within_limit": delta_within_limit,
            "gamma_within_limit": gamma_within_limit,
            "passes_all_checks": delta_within_limit and gamma_within_limit,
            "delta_utilization": abs(portfolio_delta) / self.max_portfolio_delta if self.max_portfolio_delta > 0 else 0,
            "gamma_utilization": abs(portfolio_gamma) / self.max_portfolio_gamma if self.max_portfolio_gamma > 0 else 0,
        }
    
    def calculate_var(
        self,
        portfolio_delta: float,
        spot_price: float,
        volatility: float,
        time_horizon: float = 1.0
    ) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Uses delta-normal method for VaR calculation.
        
        Args:
            portfolio_delta: Portfolio delta
            spot_price: Current spot price
            volatility: Annualized volatility
            time_horizon: Time horizon in days
        
        Returns:
            VaR value
        """
        # Convert time horizon to years
        time_years = time_horizon / 365.0
        
        # Calculate portfolio value sensitivity to price changes
        portfolio_value = portfolio_delta * spot_price
        
        # Daily volatility
        daily_vol = volatility * np.sqrt(time_years)
        
        # Z-score for confidence level
        z_score = np.abs(np.percentile(np.random.standard_normal(10000), 
                                       (1 - self.var_confidence) * 100))
        
        # VaR calculation
        var = portfolio_value * daily_vol * z_score
        
        return abs(var)
    
    def calculate_expected_shortfall(
        self,
        portfolio_delta: float,
        spot_price: float,
        volatility: float,
        time_horizon: float = 1.0,
        num_simulations: int = 10000
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            portfolio_delta: Portfolio delta
            spot_price: Current spot price
            volatility: Annualized volatility
            time_horizon: Time horizon in days
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            Expected shortfall value
        """
        # Convert time horizon to years
        time_years = time_horizon / 365.0
        
        # Daily volatility
        daily_vol = volatility * np.sqrt(time_years)
        
        # Portfolio value
        portfolio_value = portfolio_delta * spot_price
        
        # Monte Carlo simulation
        returns = np.random.normal(0, daily_vol, num_simulations)
        losses = -portfolio_value * returns
        
        # Calculate VaR threshold
        var_threshold = np.percentile(losses, self.var_confidence * 100)
        
        # Expected shortfall: average of losses beyond VaR
        tail_losses = losses[losses > var_threshold]
        expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else 0
        
        return expected_shortfall
    
    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """
        Calculate Sharpe ratio from returns.
        
        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    def calculate_maximum_drawdown(self, equity_curve: List[float]) -> Dict[str, float]:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: List of portfolio values over time
        
        Returns:
            Dictionary with drawdown metrics
        """
        if len(equity_curve) < 2:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "peak_value": equity_curve[0] if equity_curve else 0.0,
                "trough_value": equity_curve[0] if equity_curve else 0.0,
            }
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown
        drawdown = equity_array - running_max
        max_drawdown = np.min(drawdown)
        
        # Find peak and trough
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(running_max[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        peak_value = equity_array[peak_idx]
        trough_value = equity_array[max_dd_idx]
        
        max_drawdown_pct = (max_drawdown / peak_value * 100) if peak_value > 0 else 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown_pct,
            "peak_value": peak_value,
            "trough_value": trough_value,
            "peak_index": peak_idx,
            "trough_index": max_dd_idx,
        }
    
    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss_price: float
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            capital: Available capital
            risk_per_trade: Risk per trade as fraction (e.g., 0.02 = 2%)
            entry_price: Entry price for position
            stop_loss_price: Stop loss price
        
        Returns:
            Position size
        """
        risk_amount = capital * risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0.0
        
        position_size = risk_amount / price_risk
        
        # Apply maximum position size limit
        position_size = min(position_size, self.max_position_size)
        
        return position_size
