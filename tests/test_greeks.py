"""
Tests for Options Greeks Calculator.
"""

import unittest
from datetime import datetime, timedelta
from gamma_neutral.core.greeks import OptionsGreeksCalculator


class TestOptionsGreeksCalculator(unittest.TestCase):
    """Test cases for OptionsGreeksCalculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = OptionsGreeksCalculator(risk_free_rate=0.0)
    
    def test_calculate_greeks_call(self):
        """Test Greeks calculation for call option."""
        greeks = self.calculator.calculate_greeks(
            spot_price=50000.0,
            strike_price=52000.0,
            time_to_expiry=30/365.0,
            volatility=0.8,
            option_type="call"
        )
        
        # Check that all Greeks are present
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("theta", greeks)
        self.assertIn("rho", greeks)
        
        # Call delta should be between 0 and 1
        self.assertGreaterEqual(greeks["delta"], 0)
        self.assertLessEqual(greeks["delta"], 1)
        
        # Gamma should be positive
        self.assertGreater(greeks["gamma"], 0)
    
    def test_calculate_greeks_put(self):
        """Test Greeks calculation for put option."""
        greeks = self.calculator.calculate_greeks(
            spot_price=50000.0,
            strike_price=48000.0,
            time_to_expiry=30/365.0,
            volatility=0.8,
            option_type="put"
        )
        
        # Put delta should be between -1 and 0
        self.assertLessEqual(greeks["delta"], 0)
        self.assertGreaterEqual(greeks["delta"], -1)
        
        # Gamma should be positive for both calls and puts
        self.assertGreater(greeks["gamma"], 0)
    
    def test_calculate_greeks_at_expiry(self):
        """Test Greeks at expiration."""
        # ATM option at expiry
        greeks = self.calculator.calculate_greeks(
            spot_price=50000.0,
            strike_price=50000.0,
            time_to_expiry=0.0,
            volatility=0.8,
            option_type="call"
        )
        
        # At expiry, gamma, vega, theta should be zero
        self.assertEqual(greeks["gamma"], 0.0)
        self.assertEqual(greeks["vega"], 0.0)
        self.assertEqual(greeks["theta"], 0.0)
    
    def test_calculate_portfolio_gamma(self):
        """Test portfolio gamma calculation."""
        positions = [
            {
                "spot_price": 50000.0,
                "strike_price": 52000.0,
                "time_to_expiry": 30/365.0,
                "volatility": 0.8,
                "option_type": "call",
                "quantity": 10.0
            },
            {
                "spot_price": 50000.0,
                "strike_price": 48000.0,
                "time_to_expiry": 30/365.0,
                "volatility": 0.8,
                "option_type": "put",
                "quantity": -8.0
            }
        ]
        
        portfolio_gamma = self.calculator.calculate_portfolio_gamma(positions)
        
        # Portfolio gamma should be non-zero
        self.assertNotEqual(portfolio_gamma, 0.0)
    
    def test_calculate_portfolio_delta(self):
        """Test portfolio delta calculation."""
        positions = [
            {
                "spot_price": 50000.0,
                "strike_price": 52000.0,
                "time_to_expiry": 30/365.0,
                "volatility": 0.8,
                "option_type": "call",
                "quantity": 10.0
            }
        ]
        
        portfolio_delta = self.calculator.calculate_portfolio_delta(positions)
        
        # Portfolio delta should be positive for long calls
        self.assertGreater(portfolio_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
