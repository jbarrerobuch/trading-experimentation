"""
Gamma Neutral Strategy Implementation.

This module implements a gamma neutral trading strategy that uses options
and perpetual futures to maintain a gamma neutral portfolio while generating
returns from volatility and time decay.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from ..core.greeks import OptionsGreeksCalculator
from ..core.hedging import PerpetualFuturesHedger
from ..core.portfolio import PortfolioTracker, Position


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GammaNeutralStrategy:
    """
    Implements a gamma neutral trading strategy.
    
    The strategy:
    1. Maintains a gamma neutral portfolio by balancing options positions
    2. Uses perpetual futures for delta hedging
    3. Dynamically rebalances to maintain neutrality
    4. Generates returns from theta decay and volatility changes
    """
    
    def __init__(
        self,
        target_gamma: float = 0.0,
        gamma_tolerance: float = 0.1,
        delta_tolerance: float = 0.05,
        rebalance_frequency: int = 3600,  # seconds
        risk_free_rate: float = 0.0,
        min_rebalance_threshold: float = 0.1,
        transaction_cost: float = 0.0005
    ):
        """
        Initialize the gamma neutral strategy.
        
        Args:
            target_gamma: Target gamma for the portfolio
            gamma_tolerance: Maximum acceptable deviation from target gamma
            delta_tolerance: Maximum acceptable delta exposure
            rebalance_frequency: Time between rebalance checks (seconds)
            risk_free_rate: Risk-free rate for options pricing
            min_rebalance_threshold: Minimum deviation to trigger rebalance
            transaction_cost: Transaction cost as a fraction
        """
        self.target_gamma = target_gamma
        self.gamma_tolerance = gamma_tolerance
        self.delta_tolerance = delta_tolerance
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize components
        self.greeks_calculator = OptionsGreeksCalculator(risk_free_rate)
        self.hedger = PerpetualFuturesHedger(min_rebalance_threshold, transaction_cost)
        self.portfolio = PortfolioTracker()
        
        # State tracking
        self.last_rebalance_time = None
        self.rebalance_history: List[Dict] = []
    
    def add_option_position(
        self,
        quantity: float,
        entry_price: float,
        strike_price: float,
        expiry_date: datetime,
        option_type: str,
        volatility: float
    ) -> str:
        """
        Add an options position to the portfolio.
        
        Args:
            quantity: Number of contracts (positive for long, negative for short)
            entry_price: Entry price of the option
            strike_price: Strike price
            expiry_date: Expiration date
            option_type: "call" or "put"
            volatility: Implied volatility
        
        Returns:
            Position ID
        """
        position = Position(
            position_type="option",
            quantity=quantity,
            entry_price=entry_price,
            strike_price=strike_price,
            expiry_date=expiry_date,
            option_type=option_type,
            volatility=volatility
        )
        position_id = self.portfolio.add_position(position)
        logger.info(f"Added option position: {position_id}")
        return position_id
    
    def calculate_portfolio_greeks(self, spot_price: float) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            spot_price: Current spot price of underlying
        
        Returns:
            Dictionary with portfolio Greeks
        """
        positions = self.portfolio.get_options_for_greeks_calculation(spot_price)
        
        if not positions:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
            }
        
        portfolio_delta = self.greeks_calculator.calculate_portfolio_delta(positions)
        portfolio_gamma = self.greeks_calculator.calculate_portfolio_gamma(positions)
        
        # Calculate vega and theta by summing individual positions
        total_vega = 0.0
        total_theta = 0.0
        
        for position in positions:
            greeks = self.greeks_calculator.calculate_greeks(
                spot_price=position["spot_price"],
                strike_price=position["strike_price"],
                time_to_expiry=position["time_to_expiry"],
                volatility=position["volatility"],
                option_type=position["option_type"]
            )
            total_vega += greeks["vega"] * position["quantity"]
            total_theta += greeks["theta"] * position["quantity"]
        
        # Add futures delta
        futures_positions = self.portfolio.get_futures_positions()
        futures_delta = sum(p.quantity for p in futures_positions)
        total_delta = portfolio_delta + futures_delta
        
        return {
            "delta": total_delta,
            "gamma": portfolio_gamma,
            "vega": total_vega,
            "theta": total_theta,
            "options_delta": portfolio_delta,
            "futures_delta": futures_delta,
        }
    
    def check_rebalance_needed(self, spot_price: float) -> Dict[str, bool]:
        """
        Check if rebalancing is needed based on current Greeks.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            Dictionary indicating which rebalances are needed
        """
        greeks = self.calculate_portfolio_greeks(spot_price)
        
        gamma_deviation = abs(greeks["gamma"] - self.target_gamma)
        delta_deviation = abs(greeks["delta"])
        
        needs_gamma_rebalance = gamma_deviation > self.gamma_tolerance
        needs_delta_rebalance = delta_deviation > self.delta_tolerance
        
        # Check time-based rebalance
        needs_time_rebalance = False
        if self.last_rebalance_time:
            time_since_rebalance = (datetime.now() - self.last_rebalance_time).total_seconds()
            needs_time_rebalance = time_since_rebalance >= self.rebalance_frequency
        
        return {
            "needs_gamma_rebalance": needs_gamma_rebalance,
            "needs_delta_rebalance": needs_delta_rebalance,
            "needs_time_rebalance": needs_time_rebalance,
            "needs_any_rebalance": needs_gamma_rebalance or needs_delta_rebalance or needs_time_rebalance,
            "gamma_deviation": gamma_deviation,
            "delta_deviation": delta_deviation,
        }
    
    def rebalance(self, spot_price: float) -> Dict:
        """
        Rebalance the portfolio to maintain gamma neutrality.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            Rebalancing results
        """
        logger.info(f"Starting rebalance at spot price: {spot_price}")
        
        # Calculate current Greeks
        greeks = self.calculate_portfolio_greeks(spot_price)
        
        # Calculate required hedge
        hedge_info = self.hedger.calculate_gamma_hedge(
            portfolio_gamma=greeks["gamma"],
            portfolio_delta=greeks["options_delta"],
            spot_price=spot_price,
            target_gamma=self.target_gamma
        )
        
        # Execute futures hedge for delta neutrality
        if hedge_info["rebalance_needed"] or abs(greeks["delta"]) > self.delta_tolerance:
            execution = self.hedger.execute_hedge(
                futures_position=hedge_info["futures_position"],
                spot_price=spot_price
            )
            
            # Update or create futures position in portfolio
            futures_positions = self.portfolio.get_futures_positions()
            if futures_positions:
                # Update existing position
                self.portfolio.update_position_quantity(
                    futures_positions[0].position_id,
                    hedge_info["futures_position"]
                )
            else:
                # Create new futures position
                futures_pos = Position(
                    position_type="futures",
                    quantity=hedge_info["futures_position"],
                    entry_price=spot_price
                )
                self.portfolio.add_position(futures_pos)
        else:
            execution = {
                "position_change": 0.0,
                "execution_cost": 0.0,
            }
        
        # Update rebalance time
        self.last_rebalance_time = datetime.now()
        
        # Calculate new Greeks after rebalance
        new_greeks = self.calculate_portfolio_greeks(spot_price)
        
        # Log rebalance
        rebalance_record = {
            "timestamp": self.last_rebalance_time,
            "spot_price": spot_price,
            "greeks_before": greeks,
            "greeks_after": new_greeks,
            "hedge_info": hedge_info,
            "execution": execution,
        }
        self.rebalance_history.append(rebalance_record)
        
        logger.info(f"Rebalance complete. Delta: {new_greeks['delta']:.4f}, Gamma: {new_greeks['gamma']:.4f}")
        
        return rebalance_record
    
    def get_portfolio_status(self, spot_price: float) -> Dict:
        """
        Get comprehensive portfolio status.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            Portfolio status dictionary
        """
        greeks = self.calculate_portfolio_greeks(spot_price)
        rebalance_check = self.check_rebalance_needed(spot_price)
        portfolio_summary = self.portfolio.get_portfolio_summary(spot_price)
        
        return {
            "spot_price": spot_price,
            "greeks": greeks,
            "rebalance_check": rebalance_check,
            "portfolio_summary": portfolio_summary,
            "last_rebalance": self.last_rebalance_time,
            "futures_position": self.hedger.get_current_position(),
        }
    
    def get_rebalance_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get rebalancing history.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of rebalance records
        """
        if limit:
            return self.rebalance_history[-limit:]
        return self.rebalance_history
