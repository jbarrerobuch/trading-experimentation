"""
Portfolio Tracker for managing options and futures positions.

This module tracks all positions and calculates portfolio-level metrics.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta


class Position:
    """Represents a single position (option or futures)."""
    
    def __init__(
        self,
        position_type: str,
        quantity: float,
        entry_price: float,
        strike_price: Optional[float] = None,
        expiry_date: Optional[datetime] = None,
        option_type: Optional[str] = None,
        volatility: Optional[float] = None,
        position_id: Optional[str] = None
    ):
        """
        Initialize a position.
        
        Args:
            position_type: "option" or "futures"
            quantity: Number of contracts (positive for long, negative for short)
            entry_price: Price at entry
            strike_price: Strike price (for options)
            expiry_date: Expiration date (for options)
            option_type: "call" or "put" (for options)
            volatility: Implied volatility (for options)
            position_id: Unique identifier for the position
        """
        self.position_type = position_type
        self.quantity = quantity
        self.entry_price = entry_price
        self.strike_price = strike_price
        self.expiry_date = expiry_date
        self.option_type = option_type
        self.volatility = volatility
        self.position_id = position_id or f"{position_type}_{datetime.now().timestamp()}"
        self.entry_date = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "position_type": self.position_type,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "strike_price": self.strike_price,
            "expiry_date": self.expiry_date,
            "option_type": self.option_type,
            "volatility": self.volatility,
            "entry_date": self.entry_date,
        }


class PortfolioTracker:
    """
    Tracks and manages a portfolio of options and futures positions.
    """
    
    def __init__(self):
        """Initialize the portfolio tracker."""
        self.positions: Dict[str, Position] = {}
        self.history: List[Dict] = []
    
    def add_position(self, position: Position) -> str:
        """
        Add a new position to the portfolio.
        
        Args:
            position: Position object to add
        
        Returns:
            Position ID
        """
        self.positions[position.position_id] = position
        self._log_action("add", position)
        return position.position_id
    
    def remove_position(self, position_id: str) -> bool:
        """
        Remove a position from the portfolio.
        
        Args:
            position_id: ID of position to remove
        
        Returns:
            True if removed, False if not found
        """
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            self._log_action("remove", position)
            return True
        return False
    
    def update_position_quantity(self, position_id: str, new_quantity: float) -> bool:
        """
        Update the quantity of a position.
        
        Args:
            position_id: ID of position to update
            new_quantity: New quantity
        
        Returns:
            True if updated, False if not found
        """
        if position_id in self.positions:
            old_quantity = self.positions[position_id].quantity
            self.positions[position_id].quantity = new_quantity
            self._log_action("update", self.positions[position_id], 
                           extra={"old_quantity": old_quantity})
            return True
        return False
    
    def get_options_positions(self) -> List[Position]:
        """Get all options positions."""
        return [p for p in self.positions.values() if p.position_type == "option"]
    
    def get_futures_positions(self) -> List[Position]:
        """Get all futures positions."""
        return [p for p in self.positions.values() if p.position_type == "futures"]
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a specific position by ID."""
        return self.positions.get(position_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self.positions.values())
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dictionary mapping position types to current prices
        
        Returns:
            Total portfolio value
        """
        total_value = 0.0
        for position in self.positions.values():
            if position.position_type == "futures":
                current_price = current_prices.get("futures", 0.0)
                # Futures P&L
                pnl = position.quantity * (current_price - position.entry_price)
                total_value += pnl
            else:  # options
                # Would need option pricing model for accurate valuation
                # For now, use simplified approach
                current_price = current_prices.get(f"option_{position.position_id}", 0.0)
                value = position.quantity * current_price
                total_value += value
        
        return total_value
    
    def get_portfolio_summary(self, spot_price: float) -> Dict:
        """
        Get a summary of the portfolio.
        
        Args:
            spot_price: Current spot price of underlying
        
        Returns:
            Portfolio summary dictionary
        """
        options = self.get_options_positions()
        futures = self.get_futures_positions()
        
        total_options = len(options)
        total_futures = len(futures)
        
        total_options_notional = sum(
            abs(p.quantity) * spot_price for p in options
        )
        total_futures_notional = sum(
            abs(p.quantity) * spot_price for p in futures
        )
        
        return {
            "total_positions": len(self.positions),
            "options_count": total_options,
            "futures_count": total_futures,
            "options_notional": total_options_notional,
            "futures_notional": total_futures_notional,
            "total_notional": total_options_notional + total_futures_notional,
        }
    
    def get_options_for_greeks_calculation(self, spot_price: float) -> List[Dict]:
        """
        Get options positions formatted for Greeks calculation.
        
        Args:
            spot_price: Current spot price
        
        Returns:
            List of position dictionaries for Greeks calculator
        """
        options = self.get_options_positions()
        positions = []
        
        for opt in options:
            if opt.expiry_date:
                time_to_expiry = (opt.expiry_date - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                time_to_expiry = max(0, time_to_expiry)  # Ensure non-negative
            else:
                time_to_expiry = 0.0
            
            positions.append({
                "spot_price": spot_price,
                "strike_price": opt.strike_price,
                "time_to_expiry": time_to_expiry,
                "volatility": opt.volatility or 0.5,  # Default volatility
                "option_type": opt.option_type,
                "quantity": opt.quantity,
            })
        
        return positions
    
    def _log_action(self, action: str, position: Position, extra: Optional[Dict] = None):
        """Log an action to history."""
        log_entry = {
            "timestamp": datetime.now(),
            "action": action,
            "position": position.to_dict(),
        }
        if extra:
            log_entry["extra"] = extra
        self.history.append(log_entry)
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get action history.
        
        Args:
            limit: Maximum number of entries to return (most recent first)
        
        Returns:
            List of history entries
        """
        if limit:
            return self.history[-limit:]
        return self.history
