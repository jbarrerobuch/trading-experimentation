"""
Perpetual Futures Hedging Module for gamma neutral strategy.

This module manages hedging of options positions using perpetual futures
to maintain gamma neutrality.
"""

from typing import Dict, Optional
import numpy as np


class PerpetualFuturesHedger:
    """
    Manages hedging using perpetual futures contracts.
    
    Perpetual futures are used to hedge the delta and gamma exposure from 
    options positions.
    """
    
    def __init__(
        self,
        min_rebalance_threshold: float = 0.1,
        transaction_cost: float = 0.0005
    ):
        """
        Initialize the hedger.
        
        Args:
            min_rebalance_threshold: Minimum gamma deviation to trigger rebalancing
            transaction_cost: Transaction cost as a fraction (e.g., 0.0005 = 0.05%)
        """
        self.min_rebalance_threshold = min_rebalance_threshold
        self.transaction_cost = transaction_cost
        self.current_futures_position = 0.0
    
    def calculate_gamma_hedge(
        self,
        portfolio_gamma: float,
        portfolio_delta: float,
        spot_price: float,
        target_gamma: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate the required futures position to achieve target gamma.
        
        Since futures have gamma = 0, we can't directly hedge gamma with futures.
        Instead, we use dynamic delta hedging to manage gamma exposure.
        
        Args:
            portfolio_gamma: Current portfolio gamma
            portfolio_delta: Current portfolio delta
            spot_price: Current spot price of underlying
            target_gamma: Target gamma (typically 0 for gamma neutral)
        
        Returns:
            Dictionary with hedging information:
                - futures_position: Required futures position size
                - hedge_delta: Delta of the hedge
                - rebalance_needed: Whether rebalancing is needed
                - estimated_cost: Estimated transaction cost
        """
        # For gamma neutrality, we need to adjust options positions, not futures
        # Futures are used for delta hedging
        
        # Calculate required futures position for delta neutrality
        futures_delta = -portfolio_delta
        
        # Check if rebalancing is needed
        gamma_deviation = abs(portfolio_gamma - target_gamma)
        rebalance_needed = gamma_deviation > self.min_rebalance_threshold
        
        # Calculate position change
        position_change = futures_delta - self.current_futures_position
        
        # Estimate transaction cost
        estimated_cost = abs(position_change) * spot_price * self.transaction_cost
        
        return {
            "futures_position": futures_delta,
            "hedge_delta": futures_delta,
            "position_change": position_change,
            "rebalance_needed": rebalance_needed,
            "estimated_cost": estimated_cost,
            "gamma_deviation": gamma_deviation,
        }
    
    def calculate_dynamic_hedge_ratio(
        self,
        portfolio_gamma: float,
        spot_price: float,
        price_change: float
    ) -> float:
        """
        Calculate dynamic hedge ratio based on gamma exposure.
        
        As the spot price moves, gamma causes delta to change. This function
        calculates how much the futures position should change.
        
        Args:
            portfolio_gamma: Current portfolio gamma
            spot_price: Current spot price
            price_change: Expected or observed price change
        
        Returns:
            Required change in futures position
        """
        # Delta change due to gamma
        delta_change = portfolio_gamma * price_change
        
        # Required futures adjustment
        futures_adjustment = -delta_change
        
        return futures_adjustment
    
    def execute_hedge(
        self,
        futures_position: float,
        spot_price: float
    ) -> Dict[str, float]:
        """
        Execute the hedge by updating the futures position.
        
        Args:
            futures_position: Target futures position
            spot_price: Current spot price
        
        Returns:
            Execution details including costs
        """
        position_change = futures_position - self.current_futures_position
        cost = abs(position_change) * spot_price * self.transaction_cost
        
        # Update current position
        self.current_futures_position = futures_position
        
        return {
            "previous_position": self.current_futures_position - position_change,
            "new_position": self.current_futures_position,
            "position_change": position_change,
            "execution_cost": cost,
            "execution_price": spot_price,
        }
    
    def get_current_position(self) -> float:
        """Get the current futures position."""
        return self.current_futures_position
    
    def calculate_pnl(
        self,
        entry_price: float,
        current_price: float,
        position_size: Optional[float] = None
    ) -> float:
        """
        Calculate P&L from the futures position.
        
        Args:
            entry_price: Price at which position was entered
            current_price: Current price
            position_size: Position size (uses current position if not specified)
        
        Returns:
            Profit/Loss from the futures position
        """
        if position_size is None:
            position_size = self.current_futures_position
        
        pnl = position_size * (current_price - entry_price)
        return pnl
    
    def calculate_funding_cost(
        self,
        position_size: float,
        funding_rate: float,
        time_periods: int = 1
    ) -> float:
        """
        Calculate funding costs for perpetual futures.
        
        Args:
            position_size: Size of the position
            funding_rate: Funding rate per period (e.g., 0.0001 = 0.01%)
            time_periods: Number of funding periods
        
        Returns:
            Total funding cost/revenue
        """
        # Positive funding rate means longs pay shorts
        # Negative position (short) receives funding when rate is positive
        funding_cost = -position_size * funding_rate * time_periods
        return funding_cost
