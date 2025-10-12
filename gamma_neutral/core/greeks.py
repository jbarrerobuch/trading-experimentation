"""
Options Greeks Calculator for crypto options.

This module provides functionality to calculate the Greek values (delta, gamma, 
vega, theta, rho) for options positions using the Black-Scholes model.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Literal


class OptionsGreeksCalculator:
    """
    Calculator for options Greeks using Black-Scholes model.
    
    Greeks are measures of the sensitivity of options prices to various factors:
    - Delta: Sensitivity to underlying price changes
    - Gamma: Rate of change of delta
    - Vega: Sensitivity to volatility changes
    - Theta: Time decay
    - Rho: Sensitivity to interest rate changes
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize the Greeks calculator.
        
        Args:
            risk_free_rate: Annual risk-free interest rate (default 0.0 for crypto)
        """
        self.risk_free_rate = risk_free_rate
    
    def _calculate_d1_d2(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float
    ) -> tuple:
        """Calculate d1 and d2 parameters for Black-Scholes formula."""
        d1 = (np.log(spot_price / strike_price) + 
              (self.risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        return d1, d2
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        option_type: Literal["call", "put"] = "call"
    ) -> Dict[str, float]:
        """
        Calculate all Greeks for an option.
        
        Args:
            spot_price: Current price of the underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration in years
            volatility: Implied volatility (annualized)
            option_type: Type of option ("call" or "put")
        
        Returns:
            Dictionary containing delta, gamma, vega, theta, and rho values
        """
        if time_to_expiry <= 0:
            return self._calculate_greeks_at_expiry(spot_price, strike_price, option_type)
        
        d1, d2 = self._calculate_d1_d2(spot_price, strike_price, time_to_expiry, volatility)
        
        # Calculate Greeks
        if option_type == "call":
            delta = norm.cdf(d1)
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                    self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
            rho = strike_price * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                    self.risk_free_rate * strike_price * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2))
            rho = -strike_price * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)
        
        # Gamma and Vega are the same for calls and puts
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry)
        
        # Convert theta to per-day value (divide by 365)
        theta_per_day = theta / 365.0
        
        # Convert vega to per 1% volatility change
        vega_per_pct = vega / 100.0
        
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega_per_pct,
            "theta": theta_per_day,
            "rho": rho,
        }
    
    def _calculate_greeks_at_expiry(
        self,
        spot_price: float,
        strike_price: float,
        option_type: Literal["call", "put"]
    ) -> Dict[str, float]:
        """Calculate Greeks at expiration (time_to_expiry = 0)."""
        if option_type == "call":
            delta = 1.0 if spot_price > strike_price else 0.0
        else:
            delta = -1.0 if spot_price < strike_price else 0.0
        
        return {
            "delta": delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0,
        }
    
    def calculate_portfolio_gamma(
        self,
        positions: list
    ) -> float:
        """
        Calculate the total gamma of a portfolio of options.
        
        Args:
            positions: List of position dictionaries containing:
                - spot_price: Current underlying price
                - strike_price: Strike price
                - time_to_expiry: Time to expiration (years)
                - volatility: Implied volatility
                - option_type: "call" or "put"
                - quantity: Number of contracts (positive for long, negative for short)
        
        Returns:
            Total portfolio gamma
        """
        total_gamma = 0.0
        for position in positions:
            greeks = self.calculate_greeks(
                spot_price=position["spot_price"],
                strike_price=position["strike_price"],
                time_to_expiry=position["time_to_expiry"],
                volatility=position["volatility"],
                option_type=position["option_type"]
            )
            total_gamma += greeks["gamma"] * position["quantity"]
        
        return total_gamma
    
    def calculate_portfolio_delta(
        self,
        positions: list
    ) -> float:
        """
        Calculate the total delta of a portfolio of options.
        
        Args:
            positions: List of position dictionaries (same format as calculate_portfolio_gamma)
        
        Returns:
            Total portfolio delta
        """
        total_delta = 0.0
        for position in positions:
            greeks = self.calculate_greeks(
                spot_price=position["spot_price"],
                strike_price=position["strike_price"],
                time_to_expiry=position["time_to_expiry"],
                volatility=position["volatility"],
                option_type=position["option_type"]
            )
            total_delta += greeks["delta"] * position["quantity"]
        
        return total_delta
