"""
Tests for Gamma Neutral Strategy.
"""

import unittest
from datetime import datetime, timedelta
from gamma_neutral.strategies.gamma_neutral import GammaNeutralStrategy


class TestGammaNeutralStrategy(unittest.TestCase):
    """Test cases for GammaNeutralStrategy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.strategy = GammaNeutralStrategy(
            target_gamma=0.0,
            gamma_tolerance=0.1,
            delta_tolerance=0.05
        )
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.target_gamma, 0.0)
        self.assertEqual(self.strategy.gamma_tolerance, 0.1)
        self.assertEqual(self.strategy.delta_tolerance, 0.05)
    
    def test_add_option_position(self):
        """Test adding an option position."""
        expiry_date = datetime.now() + timedelta(days=30)
        position_id = self.strategy.add_option_position(
            quantity=10.0,
            entry_price=2500.0,
            strike_price=52000.0,
            expiry_date=expiry_date,
            option_type="call",
            volatility=0.8
        )
        
        # Check that position was added
        self.assertIsNotNone(position_id)
        positions = self.strategy.portfolio.get_options_positions()
        self.assertEqual(len(positions), 1)
    
    def test_calculate_portfolio_greeks(self):
        """Test portfolio Greeks calculation."""
        expiry_date = datetime.now() + timedelta(days=30)
        self.strategy.add_option_position(
            quantity=10.0,
            entry_price=2500.0,
            strike_price=52000.0,
            expiry_date=expiry_date,
            option_type="call",
            volatility=0.8
        )
        
        spot_price = 50000.0
        greeks = self.strategy.calculate_portfolio_greeks(spot_price)
        
        # Check that Greeks are calculated
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("theta", greeks)
        
        # For long calls, delta should be positive
        self.assertGreater(greeks["options_delta"], 0.0)
    
    def test_check_rebalance_needed(self):
        """Test rebalance check."""
        expiry_date = datetime.now() + timedelta(days=30)
        self.strategy.add_option_position(
            quantity=10.0,
            entry_price=2500.0,
            strike_price=52000.0,
            expiry_date=expiry_date,
            option_type="call",
            volatility=0.8
        )
        
        spot_price = 50000.0
        check = self.strategy.check_rebalance_needed(spot_price)
        
        # Check that all fields are present
        self.assertIn("needs_gamma_rebalance", check)
        self.assertIn("needs_delta_rebalance", check)
        self.assertIn("needs_any_rebalance", check)
    
    def test_rebalance(self):
        """Test portfolio rebalancing."""
        expiry_date = datetime.now() + timedelta(days=30)
        self.strategy.add_option_position(
            quantity=10.0,
            entry_price=2500.0,
            strike_price=52000.0,
            expiry_date=expiry_date,
            option_type="call",
            volatility=0.8
        )
        
        spot_price = 50000.0
        result = self.strategy.rebalance(spot_price)
        
        # Check rebalance result structure
        self.assertIn("timestamp", result)
        self.assertIn("spot_price", result)
        self.assertIn("greeks_before", result)
        self.assertIn("greeks_after", result)
        
        # After rebalance, delta should be closer to zero
        delta_after = abs(result["greeks_after"]["delta"])
        delta_before = abs(result["greeks_before"]["delta"])
        self.assertLessEqual(delta_after, delta_before + 0.1)
    
    def test_get_portfolio_status(self):
        """Test getting portfolio status."""
        expiry_date = datetime.now() + timedelta(days=30)
        self.strategy.add_option_position(
            quantity=10.0,
            entry_price=2500.0,
            strike_price=52000.0,
            expiry_date=expiry_date,
            option_type="call",
            volatility=0.8
        )
        
        spot_price = 50000.0
        status = self.strategy.get_portfolio_status(spot_price)
        
        # Check status structure
        self.assertIn("spot_price", status)
        self.assertIn("greeks", status)
        self.assertIn("rebalance_check", status)
        self.assertIn("portfolio_summary", status)
        
        # Portfolio should have 1 options position
        self.assertEqual(status["portfolio_summary"]["options_count"], 1)


if __name__ == "__main__":
    unittest.main()
